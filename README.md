# MachineLearning_StockPicker_Pipeline
ML pipeline with feature generation for predicting outperforming stocks in index.
This is not a trading system, it an equity performance classifier based on self generated features based purely on price data, so effectively a momentum factor model. Once tuned and automated, this could be used as a module in a full trading system that incorporates multiple factors (think Fama French) as well as fundamental data, sentiment data, volume, analyst revisions etc etc.

This is the first cut of a basic pipeline for the purpose of getting to know sklearn. The data munging is intended to be run straight through, but some control / breaks should be applied to 'pipeline control'.
As far as stock picking / and *good* feature genration goes, nothing here is optimised. There is a big TODO list for automation, but the bare bones are here to try various indicators and classifiers with reasonably flexible inputs.
