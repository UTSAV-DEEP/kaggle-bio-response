
all: reports/figures/exploratory.png models/random_forest.model

clean:
   rm -f data/raw/*.csv
   rm -f data/processed/*.pickle
   rm -f data/processed/*.xlsx
   rm -f reports/figures/*.png
   rm -f models/*.model

models/random_forest.model: data/processed/processed.pickle
   python src/models/train_model.py $< $@

test: all
   pytest src

.PHONY: all clean test