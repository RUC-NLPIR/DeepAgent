
import yaml
from typing import Dict, Any, List, Tuple

import alfworld
import alfworld.agents.environment


def _get_environment(env_type: str):
    """Return the ALFWorld environment class based on type in config."""
    if env_type == 'AlfredTWEnv':
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        return AlfredTWEnv
    elif env_type == 'AlfredThorEnv':
        from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
        return AlfredThorEnv
    elif env_type == 'AlfredHybrid':
        from alfworld.agents.environment.alfred_hybrid import AlfredHybrid
        return AlfredHybrid
    else:
        raise NotImplementedError(f"Environment {env_type} is not implemented.")


class ALFWorldEnvWrapper:
    """
    Thin wrapper around ALFWorld env for single-agent tool-call integration.

    - step_action(name, arguments) -> returns (observation, reward, done)
    - reset() -> initial observation
    """

    def __init__(self, batch_size: int, config_path: str = './config/alfworld_config.yaml', split: str = 'eval_out_of_distribution'):
        with open(config_path, 'r', encoding='utf-8') as reader:
            self._config = yaml.safe_load(reader)
        env_class = _get_environment(self._config["env"]["type"])
        self._batch_size = batch_size
        self._env = env_class(self._config, train_eval=split)
        self._env = self._env.init_env(batch_size=batch_size)
        self._closed = False

    def close(self):
        """Properly close the environment and clean up resources"""
        if self._closed:
            return
        
        try:
            # Call close method if it exists
            if hasattr(self._env, 'close'):
                self._env.close()
            
            # For AlfredThorEnv, also try to stop individual environments
            if hasattr(self._env, 'envs'):
                for env in self._env.envs:
                    if hasattr(env, 'env') and hasattr(env.env, 'stop'):
                        try:
                            env.env.stop()
                            print(f"Environment {env} stopped successfully")
                        except Exception as e:
                            print(f"Warning: Failed to stop environment: {e}")
            
            self._closed = True
        except Exception as e:
            print(f"Warning: Error during environment cleanup: {e}")
            self._closed = True

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.close()
        except Exception:
            # Ignore errors during cleanup in destructor
            pass

    @staticmethod
    def _process_observation(ob: str) -> str:
        # if ob.startswith('You arrive at loc '):
        #     ob = ob[ob.find('. ')+2:]
        return ob

    def reset(self) -> str:
        all_obs, _info = self._env.reset()
        initial_obs_list = []
        for i, ob in enumerate(all_obs):
            # ob = '\n'.join(ob.split('\n\n')[1:])
            ob_text = self._process_observation(ob)
            initial_obs_list.append(ob_text)
        return initial_obs_list

    @staticmethod
    def _format_action_from_tool_call(name: str, arguments: Dict[str, Any]) -> str:
        """Map a tool-call (name, arguments) to an ALFWorld textual action."""
        name = (name or '').strip().lower()

        def arg(key: str) -> str:
            val = arguments.get(key, '')
            return str(val).strip()

        if name == 'goto':
            recep = arg('recep')
            return f"go to {recep}"
        if name == 'take':
            obj = arg('obj')
            from_recep = arg('from')
            if obj and from_recep:
                return f"take {obj} from {from_recep}"
            elif obj:
                return f"take {obj}"
        if name == 'move':
            obj = arg('obj')
            to_recep = arg('to')
            if obj and to_recep:
                return f"move {obj} to {to_recep}"
            elif obj:
                return f"drop {obj}"
        if name == 'open':
            recep = arg('recep')
            return f"open {recep}"
        if name == 'clean':
            obj = arg('obj')
            with_recep = arg('with')
            if obj and with_recep:
                return f"clean {obj} with {with_recep}"
            elif obj:
                return f"clean {obj}"
        if name == 'heat':
            obj = arg('obj')
            with_recep = arg('with')
            if obj and with_recep:
                return f"heat {obj} with {with_recep}"
            elif obj:
                return f"heat {obj}"
        if name == 'cool':
            obj = arg('obj')
            with_recep = arg('with')
            if obj and with_recep:
                return f"cool {obj} with {with_recep}"
            elif obj:
                return f"cool {obj}"
        if name == 'use':
            obj = arg('obj')
            return f"use {obj}"
        if name == 'look':
            return "look"
        # Fallback to free-form action if provided
        free = arg('action') or arguments.get('raw', '') or ''
        return str(free) if free else "look"

    def step_action(self, index: int, name: str, arguments: Dict[str, Any]) -> Tuple[str, int, bool]:
        """
        Step the environment with a tool-call style action.
        Returns (observation, reward, done, action_text)
        """
        all_actions = [""] * self._batch_size
        all_actions[index] = self._format_action_from_tool_call(name, arguments)

        observation, reward, done, info = self._env.step(all_actions)
        ob_text = self._process_observation(observation[index])
        reward = reward[index]
        done = bool(done[index])
        # print(observation)
        # print(reward)
        # print(done)
        # info = info[index]
        return ob_text, reward, done



def get_alfworld_function_definitions() -> List[Dict[str, Any]]:
    """
    Return OpenAI-function style schemas for the available embodied actions in TextWorld.
    These schemas are used in the prompt to guide tool calls.
    """
    return [
        {
            "name": "goto",
            "description": "Navigate to a specific receptacle or location in the environment. This action moves you to the target location so you can interact with objects there. You must be at the receptacle's location to interact with it. After successfully going to a location, you'll see 'You arrive at [location]' confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recep": {
                        "type": "string",
                        "description": "Name of the target receptacle or location (e.g., 'cabinet 1', 'fridge 1', 'countertop 2')."
                    }
                },
                "required": ["recep"]
            }
        },
        {
            "name": "take",
            "description": "Pick up an object from a receptacle or surface. This action allows you to grab items like food, utensils, containers, and other objects from their current location. Make sure the receptacle is open if the object is inside (cabinets, drawers, fridge, microwave, safe). You can only hold one object at a time - you must move the current object before taking another. After successfully taking an object, you'll see 'You pick up [object]' confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "obj": {
                        "type": "string",
                        "description": "Name of the object to pick up (e.g., 'apple 1', 'mug 2', 'knife 1')."
                    },
                    "from": {
                        "type": "string",
                        "description": "Name of the source receptacle or surface (e.g., 'countertop 1', 'cabinet 4')."
                    }
                },
                "required": ["obj", "from"]
            }
        },
        {
            "name": "move",
            "description": "Move an object to a target receptacle or surface. This action places items in their destination location, such as putting food in the fridge or utensils in cabinets. You can only move one object at a time, and you must have the object in your inventory first. (taken previously). Some receptacles need to be opened first (cabinets, drawers, fridge, microwave, safe). After successfully moving an object, you'll see 'You move [object] to [location]' confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "obj": {
                        "type": "string",
                        "description": "Name of the object to move (e.g., 'apple 1', 'mug 1', 'knife 1')."
                    },
                    "to": {
                        "type": "string", 
                        "description": "Name of the target receptacle or surface (e.g., 'fridge 1', 'cabinet 6', 'countertop 2')."
                    }
                },
                "required": ["obj", "to"]
            }
        },
        {
            "name": "open",
            "description": "Open a closed receptacle to access its contents. This action is necessary before you can see or interact with objects inside containers like cabinets, drawers, fridge, microwave, or safe. You must be at the receptacle's location to open it. After successfully opening a receptacle, you'll see 'The [receptacle] is open' confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recep": {
                        "type": "string",
                        "description": "Name of the receptacle to open (e.g., 'cabinet 2', 'fridge 1', 'microwave 1')."
                    }
                },
                "required": ["recep"]
            }
        },
        {
            "name": "clean",
            "description": "Clean an object using a cleaning receptacle. This action washes or cleans items like dishes, utensils, or food items. The most common cleaning receptacle is the sinkbasin. You must have the object in your inventory first (use 'take' action). After successfully cleaning an object, you'll see 'You clean [object] using [receptacle]' confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "obj": {
                        "type": "string",
                        "description": "Name of the object to clean (e.g., 'mug 1', 'knife 1')."
                    },
                    "with": {
                        "type": "string",
                        "description": "Receptacle used for cleaning (e.g., 'sinkbasin 1')."
                    }
                },
                "required": ["obj", "with"]
            }
        },
        {
            "name": "heat",
            "description": "Heat an object using a heating receptacle. This action warms up items like food or containers to make them hot. You must have the object in your inventory first (use 'take' action). The microwave is the primary heating location. If you get 'Nothing happens', make sure you're at the microwave location and have the object in your inventory. After successfully heating an object, you'll see 'You heat [object] using [receptacle]' confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "obj": {
                        "type": "string",
                        "description": "Name of the object to heat (e.g., 'apple 1', 'mug 1')."
                    },
                    "with": {
                        "type": "string",
                        "description": "Receptacle used for heating (e.g., 'microwave 1')."
                    }
                },
                "required": ["obj", "with"]
            }
        },
        {
            "name": "cool",
            "description": "Cool an object using a cooling receptacle. This action chills items like food or containers to make them cool. The most common cooling receptacle is the fridge. You must have the object in your inventory first (use 'take' action). The fridge is the primary cooling location. If you get 'Nothing happens', make sure you're at the fridge location and have the object in your inventory. After successfully cooling an object, you'll see 'You cool [object] using [receptacle]' confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "obj": {
                        "type": "string",
                        "description": "Name of the object to cool (e.g., 'lettuce 1', 'mug 2')."
                    },
                    "with": {
                        "type": "string",
                        "description": "Receptacle used for cooling (e.g., 'fridge 1')."
                    }
                },
                "required": ["obj", "with"]
            }
        },
        {
            "name": "use",
            "description": "Use an object or receptacle. You must be at the object's location to use it. The commonly used object is desklamps for lighting objects, which helps you to examine objects. After successfully using an desklamp, you'll see 'You turn on [object]' confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "obj": {
                        "type": "string",
                        "description": "Name of the object or receptacle to use (e.g., 'desklamp 1')."
                    }
                },
                "required": ["obj"]
            }
        },
        # {
        #     "name": "look",
        #     "description": "Observe the current environment and return visible objects and receptacles. This action is always available and helps you understand what objects are present at your current location. This is useful when you arrive at a new location or need to verify what's available.",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {},
        #         "required": []
        #     }
        # }
    ]