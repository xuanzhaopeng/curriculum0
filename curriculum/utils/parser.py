import os


def parse_reward_func(config_reward_function: str) -> tuple[str, str]:
    if config_reward_function is not None:
        if ":" not in config_reward_function:
            reward_function_name = "main"
        else:
            reward_function, reward_function_name = config_reward_function.rsplit(":", maxsplit=1)

        if os.path.exists(reward_function):
            reward_function = os.path.abspath(reward_function)
        else:
            reward_function = None
        return reward_function, reward_function_name
    return None, None

def parse_formt_prompt_path(config_data_format_prompt) -> str:
    format_prompt = None
    if config_data_format_prompt is not None:
        if os.path.exists(config_data_format_prompt):  # ray job uses absolute path
            format_prompt = os.path.abspath(config_data_format_prompt)
        else:
            format_prompt = None
    return format_prompt