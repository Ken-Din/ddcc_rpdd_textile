# DDCC/RPDD Textile Return Optimization

Python implementation of a Decision Support System for textile 
product return disposition using min-cost flow optimization.

## Paper
"DDCC/RPDD: A Decision Support Framework for Reverse Logistics"
Target journal: Decision Support Systems (Elsevier)

## Requirements
pip install -r requirements.txt

## Quick Start
python main.py

## Experiments
python experiments/scalability.py
python experiments/sensitivity.py  
python experiments/information_loss.py

## Tests
python tests/test_consistency.py

## Expected Results (I=100, seed=42)
- Total utility: 5136.94
- Binding: Repair, Refurbishing, Recycling, Donation
- Disposal: 3 units (capacity-constrained)
- Consistency tests: 4/4 PASS
