# MachineLearning_StockPicker_Pipeline
ML pipeline with feature generation for predicting outperforming stocks in index.
This is not a self contained trading system, it is an equity performance classifier based on self generated features based purely on price data, so effectively a momentum factor model. Once tuned and automated, this could be used as a module in a full trading system that incorporates multiple factors (think Fama French) as well as fundamental data, sentiment data, volume, analyst revisions etc etc.

This is the first cut of a basic pipeline for the purpose of getting to know sklearn and should allow you to get a sense of time complexity limitation and effectiveness of the various sklearn estimators on a reasonably sized, and very noisy data set. 

The data munging is intended to be run straight through, but some control / breaks should be applied to 'pipeline control'.
As far as stock picking / and *good* feature genration goes, nothing here is optimised. There is a big TODO list for automation, but the bare bones are here to try various indicators and classifiers with reasonably flexible inputs.

There is a TA-lib dependency in some of the technical analysis features: I suggest going here http://mrjbq7.github.io/ta-lib/func.html

Finally, I used a bloomberg download for my dataset, but it's just adjusted closing price with dates in left most column and stock ticker above each price column. I pulled this out of excel, so if you are say pulling directly from am api, be aware that I convert the dataframe indices into timestamps. Once you have the data in that format is should run smoothly.

