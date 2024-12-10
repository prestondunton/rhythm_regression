# Rhythm Regression

## Notes

I believe that `librosa.load()`  takes so long to load files because it is using `ffmpeg` to convert my `.m4a` files into something readable by `librosa`.

## Virtual Environment

This project's virtual environment is managed using `pipenv`, which is a combination of `pip` and `virtualenv`.

To install all packages needed for the repository, run ```pipenv install```

To enter the virtual enviroment run ```pipenv shell```

## Testing

To test the code for the `rhythm_regression` package, the `test` package is provided.  To run all the tests, first make sure that the virtual environment is activated, and then in the topmost directory run the command:

```python -m unittest discover```

This uses `unittest`'s test discovering which runs files that match the pattern `test*.py`.