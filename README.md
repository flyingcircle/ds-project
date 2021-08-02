

Since the data is quite a lot, I added `data/` to the `.gitignore`. So create a `data/` dir and then add the bus data csv files to it:

`transit_daily.csv`

To run the code you'll also need a `results/` dir. Then you should be able to run

```
cd src
python3 run.py
```

This will run the svm and sgd models and place results in the `results` dir. 

You may need to install some additional python packages to get this working. I think all of the relevant packages are tracked in the `requirements.txt` but there may be a few more beyond this that are still needed.