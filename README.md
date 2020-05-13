# Invictus_Capital_Language_Detection
This repo contains the code, data and report detailing the work done to create a language detection model based on text input.

# Instructions (the instructions are included in the instructions.txt file for ease of use)
Please follow the steps below to help you set up the environment in which the model will be tested.

1. System requirements
    - Check if your Python environment is already configured by running each of the following lines of code in your terminal/command line:
        
            python3 --version
            pip3 --version
            virtualenv --version

        If these packages are already installed, skip to the next step.
        Otherwise, install Python, the pip package manager, and Virtualenv using the Homebrew package manager.
        Here is the code to do so:
        
            For MacOS:
                /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
                export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
                brew update
                brew install python  # Python 3
                sudo pip3 install -U virtualenv  # system-wide install
                
            For Windows:
                pip3 install -U pip virtualenv

    - Create a virtual environment (optional and recommended) by running the following code in your terminal/command line:
        
        For Ubuntu/MacOS:
            
            virtualenv --system-site-packages -p python3 ./venv
            source ./venv/bin/activate  # sh, bash, ksh, or zsh
            pip install --upgrade pip
            pip list  # show packages installed within the virtual environment
            deactivate  # don't exit until you're done using TensorFlow
        
        For Windows:
            
            virtualenv --system-site-packages -p python3 ./venv
            .\venv\Scripts\activate
            pip install --upgrade pip
            pip list  # show packages installed within the virtual environment
            deactivate  # don't exit until you're done using TensorFlow
            
    Inside the virtual environment, do the following:
    
    - Install the TensorFlow pip package
        
        Through the virtual environment:
            
            pip install --upgrade tensorflow
            python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
        
        System install:
            
            pip3 install --user --upgrade tensorflow  # install in $HOME
            python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

    - Install the Keras library
    
            pip install keras

    - Install the h5py library # this is to load the completed model

            pip install h5py

    - Install Pandas
    
            pip install pandas

    - Install Sklearn
    
            pip install sklearn

    - Install Numpy
    
            pip install numpy

2. Please download the following files and save them in THE SAME folder:
    
    - python_file_RUN_ME.py # main Python file that needs to be run
    - model.h5 # contains model parameters
    - model.json # contains the model architecture
    - lang_data_test.csv # (this is the file provided by the reviewer on which the model will be tested - see below *)
    - vocab.txt # this file contains the library used for training the model that's required for predictions

    PLEASE MAKE SURE THAT THE CSV FILE CONTAINING THE TEXT ON WHICH THE PREDICTIONS WILL BE MADE
    IS NAMED 'lang_data_test.csv'. THIS FILE NEEDS TO BE IN EXACTLY THE SAME FORMAT AS THE 'lang_data.csv' CSV 
    FILE PROVIDED TO ME FOR TRAINING THE MODEL ON. I HAVE PROVIDED THE TEST SET ON WHICH MY MODEL WAS TESTED.
    IT HAS BEEN NAMED 'lang_data_test.csv' AND IS INCLUDED IN THIS GITHUB FOLDER. FEEL FREE TO REPLACE THIS
    FILE WITH YOUR OWN TO SEE HOW THE MODEL PERFORMS ON A SEPARATE SET.
    
    The text file that the tester needs to save as 'lang_data_test.csv' needs to include at least a single text of each
    of the three languages (English, Afrikaans and Nederlands). There is a bug in the
    python_file_RUN_ME.py file that needs to be addressed that prevents an accuracy measure from
    being returned when there are two or fewer languages in the 'lang_data_test.csv' file.
    
3. CD (change directory) into the directory where the above files are located and run the following
code in your terminal/command line: python3 python_file_RUN_ME.py 

This step runs the Python file that first converts the test data into a format that the saved model
can interpret. The file then 

** Other related files included in this Github repo include:
    
    - LanguageDetection.ipynb: This is the Jupyter Notebook in which the initial exploratory data analysis, 
    feature engineering, model fitting and testing was done.
    - Language Detection Report.pdf: This the the report detailing the work done to clean the data and 
    to create, train and test the model  
    - notes.txt: This file contains additional notes and potential pitfalls. I tried addressing them in the report as well.
