import os

if __name__ == "__main__":

    curr_cwd = os.getcwd()

    env_vars = {
        'PYTHONPATH': "${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier".replace("$(pwd)", curr_cwd),
        'CONFIG_PATH': curr_cwd
    }

    with open('set_env_vars.env', 'w') as f:
        for k, v in env_vars.items():
            f.write(f"{k}={v};")

    # PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"

    print(env_vars)