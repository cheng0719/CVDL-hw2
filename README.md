# NCKU Computer Vision and Deep Learning Course - Homework 2

## Project Specifications
For detailed instructions on each section, please refer to the [Cvdl_HW2_Spec](https://github.com/cheng0719/Computer_Vision_and_Deep_Learning_HW2/blob/master/Cvdl_Hw2_Spec.pdf).

## Environment Setup
To ensure smooth execution of the project code, the project includes a `Pipfile`, `Pipfile.lock`, and `.python-version` for managing dependencies and the Python version. Follow the steps below to set up the virtual environment:

### Method 1: Using requirements.txt
Alternatively, you can set up the environment using the `requirements.txt` file, which lists the same dependencies as the `Pipfile`.

#### Step 1: Install Python 3.9
Ensure you are using Python 3.9, as indicated in the `.python-version file`. You can install it via your preferred method or use `pyenv`:
```
pyenv install 3.9.x
pyenv local 3.9.x
```

#### Step 2: Install Dependencies
Once you have the correct Python version, install the dependencies using the following command:
```
pip install -r requirements.txt
```

#### Step 3: Run the Project Code
After installing the dependencies, you can run the project code:
```
python main.py
```  

---

### Method 2: Using pipenv (Pipfile and Pipfile.lock)
#### Step 1: Install `pipenv`
If you don't have pipenv installed, you can install it by running:
```
pip install pipenv
```

#### Step 2: Set the Python Version
The project uses a specific Python version, as indicated in the `.python-version` file. You can set the correct Python version using one of the following methods:
- **Using `pyenv`**: Install and set the Python version specified in the `.python-version` file with:
```
pyenv install $(cat .python-version)
pyenv local $(cat .python-version)
```

- **Other methods**: If you're not using `pyenv`, ensure that you have the required Python version installed by checking `.python-version`. You can manually install the correct version from the [official Python website](https://www.python.org/downloads/) or use your preferred Python version manager.

#### Step 3: Create and Activate the Virtual Environment
After setting the Python version, install the required packages and set up the virtual environment by running:
```
pipenv install
```
This will automatically install the dependencies listed in the `Pipfile` and lock the versions specified in `Pipfile.lock`.

#### Step 4: Activate the Virtual Environment
To activate the virtual environment, run:
```
pipenv shell
```

#### Step 5: Run the Project Code
Once the virtual environment is activated, you can proceed to execute the following command to run the project:
```
python main.py
```

## Model Weights
To run the project with pre-trained model weights, you can download the model files using the following link:  

[Download ResNet50 binary classifier Model Weights](https://drive.google.com/file/d/1eBUr5FAUkSorrWFnlRb10ZvNIJy_Omvy/view?usp=sharing)  
[Download VGG19 Model Weights](https://drive.google.com/file/d/1duk9cANV4OsNElA8aYfD4ZzGnaB7HpBe/view?usp=sharing)  

After downloading the weights, place them in the appropriate directory as specified in the project code before running the model.
