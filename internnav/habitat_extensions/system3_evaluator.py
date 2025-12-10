import argparse
import json
import os
import sys
import copy
import re
import itertools
import numpy as np
import torch
import tqdm
from PIL import Image

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import Evaluator
from internnav.habitat_extensions.habitat_vln_evaluator import HabitatVLNEvaluator, DEFAULT_IMAGE_TOKEN
from internnav.agent.system3_agent import VLMNavigator, pil_to_base64
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from depth_camera_filtering import filter_depth
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions, chunk_token

@Evaluator.register('habitat_system3')
class HabitatSystem3Evaluator(HabitatVLNEvaluator):
    def __init__(self, cfg: EvalCfg):
        super().__init__(cfg)
        # System 3 specific config (could be passed via cfg or env vars)
        self.vlm_api_key = os.environ.get("VLLM_API_KEY")
        self.vlm_base_url = os.environ.get("VLLM_API_URL")
        self.vlm_model_name = os.environ.get("MODEL_NAME")

    def _run_local_eval(self) -> None:  # noqa: C901
        """
        Run local evaluation on this rank.
        Modified to include System 3 VLM Navigation Logic.
        """
        sucs, spls, oss, nes = [], [], [], []
        self.model.eval()

        # resume from previous results
        if os.path.exists(os.path.join(self.output_path, 'progress.json')):
            with open(os.path.join(self.output_path, 'progress.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    if "scene_id" not in res:
                        print("This evaluation has already finished!")
                        return (
                            torch.tensor(sucs).to(self.device),
                            torch.tensor(spls).to(self.device),
                            torch.tensor(oss).to(self.device),
                            torch.tensor(nes).to(self.device),
                            torch.tensor(len(sucs)).to(self.device),
                        )
                    if self.rank == 0:  # noqa: F405
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        nes.append(res['ne'])

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")
        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = (
                episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
            )
            print("episode start", episode_instruction)

            # Initialize System 3 Navigator
            navigator = VLMNavigator(
                user_goal=episode_instruction,
                api_key=self.vlm_api_key,
                base_url=self.vlm_base_url,
                model_name=self.vlm_model_name
            )
            current_instruction = episode_instruction # Start with the global goal

            agent_state = self.env._env.sim.get_agent_state()
            rotation = agent_state.rotation
            translation = agent_state.position
            rotation_matrix = None # Need to import quaternion or use helper if available, but let's see where it's used
            # Actually, rotation_matrix is used for transformation_matrix.
            # HabitatVLNEvaluator imports quaternion. We need to check if we can access it.
            # We imported quaternion in the original file, let's assume it's available or we can recalculate.
            # In the original code: import quaternion.
            # Since I cannot easily import 'quaternion' if it's not a standard package (it's numpy-quaternion),
            # I'll rely on what HabitatVLNEvaluator does. It imports it.
            # Let's try to import it here too.
            import quaternion

            rotation_matrix = quaternion.as_rotation_matrix(rotation)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            agent = ShortestPathFollower(self.env._env.sim, 0.25, False)

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            initial_height = self.env._env.sim.get_agent_state().position[1]

            rgb_list = []
            action_seq = []
            output_ids = None

            goal = None
            action = None
            messages = []
            local_actions = []

            done = False

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                camera_yaw = observations["compass"][0]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                # --- System 3 Update ---
                # Only update when we are about to plan a new sequence or step
                # Maybe just update every step or when action_seq is empty?
                # System 3 is "slow", so maybe not every step.
                # Let's do it when action_seq is empty and we are looking for a new high-level action.
                # Or simply: The prompt always uses `current_instruction`.
                # We update `current_instruction` using the VLM.
                
                # To save cost/time, let's only call VLM if we are about to generate a new plan (action_seq is empty)
                if len(action_seq) == 0 and goal is None: 
                     # Convert RGB to Base64
                    image_pil = Image.fromarray(rgb).convert('RGB')
                    img_b64 = pil_to_base64(image_pil)
                    
                    # Call System 3
                    # print("Calling System 3...")
                    plan = navigator.plan_next_step(img_b64)
                    if plan:
                        if plan.get("status") == "DONE":
                            print("System 3 decided goal is reached.")
                            # Force stop? Or just let System 2 decide?
                            # If System 3 says DONE, we should probably stop.
                            # action = 0 # STOP
                            # But let's just update instruction to "Stop" or something?
                            # Or we can break the loop if we trust it.
                            # For now, let's trust System 3's instruction update.
                            pass
                        
                        new_instr = plan.get("instruction")
                        if new_instr:
                            current_instruction = new_instr
                            print(f"System 3 updated instruction: {current_instruction}")

                agent_state = self.env._env.sim.get_agent_state()
                height = agent_state.position[1] - initial_height
                camera_position = np.array([x, -y, self._camera_height + height])
                tf_camera_to_episodic = (
                    self.xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30))
                    @ self.get_axis_align_matrix()
                )

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                save_dot = False
                if action == 5:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                    look_down_depth, resize_shape = self.preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                    if self.model_args.mode == 'dual_system':
                        down_observations, _, done, _ = self.env.step(5)
                        down_observations, _, done, _ = self.env.step(5)

                        look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                        depth = down_observations["depth"]
                        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                        depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                        depth = depth * 1000
                        look_down_depth, resize_shape = self.preprocess_depth_image_v2(
                            Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                            do_depth_scale=True,
                            depth_scale=1000,
                            target_height=224,
                            target_width=224,
                        )
                        look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                        look_down_depth[look_down_depth > 5.0] = 5.0

                        self.env.step(4)
                        self.env.step(4)

                info = self.env.get_metrics()

                if len(action_seq) == 0 and goal is None:
                    if action != 5:
                        sources = copy.deepcopy(self.conversation)
                        # USE CURRENT_INSTRUCTION HERE
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', current_instruction[:-1] if current_instruction.endswith('.') else current_instruction
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        # print('history_idddddddd', step_id, history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0
                    else:
                        assert action == 5
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        input_img_id = -1

                    prompt = list(self.conjunctions)[np.random.randint(len(self.conjunctions))] + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    # print('sources', step_id, sources)
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    # print('step_id', step_id, 'messages:', messages)

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]
                        pixel_goal = [int(coord[1]), int(coord[0])]

                        intrinsic_matrix = self.get_intrinsic_matrix(
                            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
                        )
                        goal = self.pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)
                        # print('before', goal, depth.shape)
                        goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3]

                        if not self.env._env.sim.pathfinder.is_navigable(np.array(goal)):
                            goal = np.array(self.env._env.sim.pathfinder.snap_point(np.array(goal)))

                        # look down --> horizontal
                        self.env.step(4)
                        self.env.step(4)

                        # Forking logic based on mode
                        if self.model_args.mode == 'system2':
                            action = agent.get_next_action(goal)
                            if action == 0:
                                goal = None
                                output_ids = None
                                action = 2  # random action
                                print('conduct a random action 2')
                                observations, _, done, _ = self.env.step(action)
                                step_id += 1
                                messages = []
                                continue
                        else:  # dual-system logic
                            local_actions = []
                            pixel_values = inputs.pixel_values
                            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

                            with torch.no_grad():
                                traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)

                            # prepocess align with navdp
                            image_dp = (
                                torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                            )
                            pix_goal_image = copy.copy(image_dp)
                            images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                            depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)
                            pix_goal_depth = copy.copy(depth_dp)
                            depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)

                            with torch.no_grad():
                                dp_actions = self.model.generate_traj(
                                    traj_latents, images_dp, depths_dp, use_async=True
                                )

                            random_choice = np.random.choice(dp_actions.shape[0])
                            if self.model_args.continuous_traj:
                                action_list = traj_to_actions(dp_actions)
                                if len(action_list) < 8:
                                    action_list += [0] * (8 - len(action_list))
                            else:
                                action_list = chunk_token(dp_actions[random_choice])

                            local_actions = action_list
                            if len(local_actions) >= 4:
                                local_actions = local_actions[:4]
                            action = local_actions[0]
                            if action == 0:
                                goal = None
                                output_ids = None
                                action = 2  # random action
                                print('conduct a random action 2')
                                observations, _, done, _ = self.env.step(action)
                                step_id += 1
                                messages = []
                                continue

                        print('predicted goal', pixel_goal, goal, flush=True)
                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif goal is not None:
                    # Forking logic based on mode
                    if self.model_args.mode == 'system2':
                        action = agent.get_next_action(goal)
                        action = action.detach().cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                        action = action[0] if hasattr(action, "__len__") else action
                    else:  # dual-system logic
                        if len(local_actions) == 0:
                            # navdp
                            local_actions = []
                            image_dp = (
                                torch.tensor(np.array(look_down_image.resize((224, 224)))).to(torch.bfloat16) / 255
                            )

                            images_dp = torch.stack([pix_goal_image, image_dp]).unsqueeze(0).to(self.device)
                            depth_dp = look_down_depth.unsqueeze(-1).to(torch.bfloat16)

                            depths_dp = torch.stack([pix_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                dp_actions = self.model.generate_traj(
                                    traj_latents, images_dp, depths_dp, use_async=True
                                )

                            random_choice = np.random.choice(dp_actions.shape[0])
                            if self.model_args.continuous_traj:
                                action_list = traj_to_actions(dp_actions)
                                if len(action_list) < 8:
                                    action_list += [0] * (8 - len(action_list))
                            else:
                                action_list = chunk_token(dp_actions[random_choice])
                            print("first action_list", action_list)

                            local_actions = action_list
                            if len(local_actions) >= 4:
                                local_actions = local_actions[:4]
                            # if len(local_actions) >= 2:
                            #     local_actions = local_actions[:2]

                            print("local_actions", local_actions)

                            action = local_actions.pop(0)
                            # navdp
                        else:
                            action = local_actions.pop(0)

                    forward_action += 1
                    print('forward_action', forward_action, flush=True)
                    if forward_action > 8:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                    if action == 0:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        local_actions = []
                        continue
                else:
                    action = 0

                if info['top_down_map'] is not None:
                    if save_dot:
                        save_raw_image = self.dot_matrix_two_dimensional(
                            save_raw_image, save_img=False, save_path=f'test_{step_id}.jpg', pixel_goal=pixel_goal
                        )
                    if self.save_video:
                        frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                        vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                # refactor: core
                if action == 5:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []

            # ---------- 3. End of episode -----------
            # Update result and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            # Write per-episode result.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
            }
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")
            if self.save_video:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(len(sucs)).to(self.device),
        )
