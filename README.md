# Robótica (Proyecto final)

> "Development of an Autonomous Visual Object Tracking System using Deep Reinforcement Learning."

This repository contains the codebase for implementing computer vision using OpenCV to detect a ball and a DQN (Deep Q-Network) agent to guide the PioneerP3DX robot in following the detected ball.

The project itself is used as the final project for the Robotics course taken as part of my [MSc studies](https://www.fi.upm.es/?id=muii) at [UPM](https://www.upm.es).

## Project Overview

The project combines robotics, computer vision, and reinforcement learning techniques to enable the PioneerP3DX robot to autonomously track and follow a ball using its onboard camera. The robot's movements are controlled by a DQN agent trained to navigate and follow the ball within a simulated environment.

## Requirements

- Python 3.9.
- Other necessary Python libraries listed in `requirements.txt`.
- CoppeliaSim (Robot simulation environment).

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/PioneerP3DX-Robotics-Project.git
    ```
2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download and install CoppeliaSim from [Coppelia Robotics](https://www.coppeliarobotics.com).

## Usage

Two main usecases are being regarded in this project: training the model and testing (demoing) it. For each case, the following steps should be followed:

### Training

1. Open CoppeliaSim and load the file `scenes/training_scene.ttt`.
2. Adjust the constants in `src/simulations/application/train_service.py`, for the training itself; and the constants in `src/simulations/domain/controlers/visual_DQN_agent.py`, for training of the actual DQN agent.
3. Run the training file `src/simulations/application/train_service.py` with the project root as the working directory.

### Demo

1. Open CoppeliaSim and load the file `scenes/demo_scene.ttt`.
2. Adjust the constants in `src/simulations/application/demo_service.py`.
3. Run the demo file `src/simulations/application/demo_service.py` with the project root as the working directory.

### Additional Services

On the directory `src/simulations/application/` other application services can be found aswell. They serve a variety of purposes which are described below:
1. `model_building_service.py`: Used to build the model architecture and save it to a file.
2. `model_transforming_service.py`: Used to transform an already TensorFlow model to a TFLite one, so predictions are performed faster (ideal for demos).
3. `statistics_service.py`: Used to generate some plots for the project's report.

## Project Structure

- `/figures`: Is the directory where the plots generated by the `statistics_service.py` are saved.
- `/logs`: Is the directory where the logs generated by the `train_service.py` are saved. A TensorBoard server can be run from the directories generated here at trainig time.
- `/models` Is the directory where the models generated by the `model_building_service.py` are saved. Also, instances of these models are generated during trainig.
- `/scenes`: Is the directory where the CoppeliaSim scenes are stored.
- `/src`: Main source code directory. Check comments on those files for more detailed information.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, feel free to open an issue or submit a pull request.