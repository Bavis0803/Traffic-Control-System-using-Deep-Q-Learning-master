# Performance Testing of SUMO Reinforcement Learning Model

This project aims to evaluate the performance of a trained reinforcement learning model for traffic light control using the SUMO (Simulation of Urban MObility) environment. The model is designed to optimize traffic flow at intersections by managing traffic light phases based on real-time traffic conditions.

## Project Structure

```
sumo-rl-performance-test
├── src
│   ├── test_performance.py      # Main logic for testing model performance
│   ├── tlsClass.py              # Traffic light system management
│   └── sumoClass.py             # SUMO simulation interface
├── data
│   └── cross.sumocfg            # SUMO configuration file
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd sumo-rl-performance-test
   ```

2. **Install Dependencies**
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **SUMO Installation**
   Make sure you have SUMO installed on your system. You can download it from [the official SUMO website](https://www.eclipse.org/sumo/).

4. **Configure SUMO Environment**
   Set the SUMO_HOME environment variable to point to your SUMO installation:
   ```bash
   export SUMO_HOME="/path/to/sumo"
   ```

## Usage

To test the performance of the trained model, run the following command:

```bash
python src/test_performance.py
```

This script will initialize the SUMO environment, load the trained model, and compare its performance against the baseline performance before training. The results will be printed to the console and can be analyzed to assess the effectiveness of the model.

## Performance Testing Process

The performance testing process involves the following steps:

1. **Initialization**: Set up the SUMO simulation environment and load the trained model.
2. **Baseline Performance**: Run the simulation without the trained model to gather baseline performance metrics.
3. **Model Performance**: Execute the simulation with the trained model and collect performance metrics.
4. **Comparison**: Analyze and compare the results from the baseline and model performance to evaluate improvements in traffic flow and efficiency.

## Conclusion

This project provides a framework for testing and evaluating the performance of reinforcement learning models in traffic light control scenarios using SUMO. By following the setup instructions and running the performance tests, users can gain insights into the effectiveness of their models and make informed decisions for further improvements.