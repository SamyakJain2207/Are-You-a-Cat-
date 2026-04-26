from scripts.train import main

# Each line = one experiment run
main(lr=1e-4, dropout=0.3)   # Run 1: baseline
main(lr=1e-3, dropout=0.3)   # Run 2: higher learning rate
main(lr=1e-4, dropout=0.5)   # Run 3: more dropout
main(lr=1e-3, dropout=0.5)   # Run 4: both changed