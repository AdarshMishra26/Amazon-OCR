1. **Install virtualenv**

    If you haven't installed `virtualenv` yet, you can install it using pip.

    ```bash
    pip install virtualenv
    ```

2. **Create a virtual environment**

    Navigate to the directory where you want to create your virtual environment and run the following command:

    ```bash
    virtualenv venv
    ```

    Here, `venv` is the name of your virtual environment. You can name it anything you like.

3. **Activate the virtual environment**

    Now that you have created a virtual environment, you need to activate it. The command to do this varies based on your operating system:

      ```bash
      venv\Scripts\activate
      ```

    Once the virtual environment is activated, your shell prompt will change to show the name of the activated virtual environment.

4. **Install packages**

    Now you can start installing and using Python packages within this isolated environment. For example:

    ```bash
    pip install requirements.txt
    ```

5. **Deactivate the virtual environment**

    Once you're done with your work, you can deactivate the virtual environment and return to your normal shell by simply typing:

    ```bash
    deactivate
    ```