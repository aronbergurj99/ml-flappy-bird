## Getting started


_The project uses some external dependencies which are listed in requirements.txt and can be installed using the following instructions._

1. Create a virtual environment 

    ```sh
    python -m venv venv
    ```
2. Activate environment
  * Mac and linux

    ```sh
    source venv/bin/activate
    ```
  * Windows

    ```sh
    source venv/Scripts/activate
    ```
3. Install requirements.txt

    ```sh
    pip install -r requirements.txt
    ```

## Training a agent
_The training script uses a function called obsorve_policy which saves a heatmaps taken n times to results/{agent-time}/plots af_

*
    ```sh
    python -m training
    ```
  
## Loading an agent
_You can load an agent by copying agent from results folder to load folder and specifing the name of the agent_

*
    ```sh
    python -m load
    ```