# movies
python download_and_save_data.py data=amazon_movies_5core
python train_tokenizer.py data=amazon_movies_5core
python tokenize_dataset.py data=amazon_movies_5core tokenizer_name=amazon_movies_5core_simple
python pretrain_HF.py seed=42 name=amazon_movies_reverse experiment=amazon_movies_default model.batch_size=64 model.epochs=10 model.train_test_split=.02 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=1000 model.randomize_order=True impl.save_intermediate_checkpoints=True
python eval.py seed=42 name=amazon_movies experiment=amazon_movies_default model.batch_size=64 model.seq_length=10 model.randomize_order=True

#small test
python train_tokenizer.py data=amazon_movies_5core data.size="_small"
python tokenize_dataset.py data=amazon_movies_5core tokenizer_name=amazon_movies_5core_small_simple data.size="_small"
python pretrain_HF.py name=amazon_movies_small experiment=amazon_movies_small model.batch_size=64 model.epochs=20 model.train_test_split=.1 model.seq_length=10 model.lr=.00001 impl.print_loss_every_nth_step=100 model.randomize_order=False
python eval.py seed=42 name=amazon_movies experiment=amazon_movies_small data.size="_small" tokenizer_name=amazon_movies_5core_small_simple model.batch_size=64 model.seq_length=10 model.randomize_order=True

