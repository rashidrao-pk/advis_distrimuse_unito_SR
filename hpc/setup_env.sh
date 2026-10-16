#!/usr/bin/env bash
# Launch an allocated compute-node tmux shell with the project environment ready.
# Use Bash explicitly: bash hpc/setup_env.sh

if [[ $- == *i* ]]; then
    # This file also serves as the tmux shell's rcfile. Skip ~/.bashrc,
    # whose micromamba initialization may use a binary for another architecture.
    if [[ ! -f /beegfs/home/mrashid/pt_312/bin/activate ]]; then
        printf 'Python environment is missing: /beegfs/home/mrashid/pt_312\n' >&2
        return 1
    fi
    source /beegfs/home/mrashid/pt_312/bin/activate
    export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages/
    if ! python --version; then
        printf 'Python could not run. Check that pt_312 supports this compute node architecture.\n' >&2
        return 1
    fi
    printf 'Environment ready in %s. Run your project commands here.\n' "$PWD"
    return
fi

set -euo pipefail
script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/setup_env.sh"
project_dir="$(dirname -- "$(dirname -- "$script_path")")"

if [[ ${1:-} != --compute && -z ${SLURM_JOB_ID:-} ]]; then
    exec srun -p epito --gres=gpu:a100:1 -J XAD --pty \
        bash --noprofile --norc "$script_path" --compute
fi

cd -- "$project_dir"
if [[ -n ${TMUX:-} ]]; then
    exec bash --noprofile --rcfile "$script_path" -i
fi

# Quote the rcfile path for the shell tmux uses to launch the command.
printf -v shell_command 'exec bash --noprofile --rcfile %q -i' "$script_path"
exec tmux new-session -s YOLO -c "$project_dir" "$shell_command"
