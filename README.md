# Welding Temperature Prediction

NTUA Master Advance Manifacturing Systems Exercise

## Installation

```bash
pip3 install virtualenv
virtualenv neural_network
source neural_network/bin/activate
pip3 install -r requirements.txt
```

## Training

For maximum temperature NN training, run this command

```bash
python3 MaxTemperatureNN_training.py
```

For time over 723 Celcius degrees NN training, run this command

```bash
python3 AboveStandardTemperatureNN_training.py
```

## Usage

For maximum temperature prediction, run this command

```bash
python3 MaxTemperatureNN_inference.py --plate_thickness 0.004 --initial_temperature 180 --heat_input 900 --electrode_velocity 0.004 --X 0.0 --Y 0.02 --Z 0.002
```

For time over 723 Celcius degrees, run this command

```bash
 python3 AboveStandardTemperatureNN_inference.py --plate_thickness 0.005 --initial_temperature 200 --heat_input 1200 --electrode_velocity 0.0035 --X 0.025 --Y 0.025 --Z 0.0025 
```
