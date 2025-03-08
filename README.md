# Ethical-AI-teachers
A ethics in AI course project on when AI tutors help students cheat.

## Setup
Make sure you're on a compute node. Run:
```
module load cuda/12.4.1 
module load gcc/9.3.0
bash setup.sh
source .venv/bin/activate
```
This installs uv and the dependencies into a virtual environment. Feel free to use conda.

## Running the Code
Running with the debug flag on will only run the first 10 tasks.
```
python main.py --model meta-llama/Llama-3.1-8B-Instruct --num_gpus 1 --greedy --debug
```
Running with the debug flag off will run all tasks.
```
python main.py --model meta-llama/Llama-3.1-8B-Instruct --num_gpus 1 --greedy
```
This will generate all the samples and evaluate them.

## Where to Modify the Code
To modify what information is included in the samples, you can probably modify [main.py](main.py#L34), and if you'd like to modify where data is saved, you can modify [codegen.py](codegen.py#L212); both links point to specific lines in the code.