"""
Utility del progetto.

Import espliciti dai sottomoduli, es.:
  from src.utils.checkpoint import load_checkpoint
  from src.utils.config import load_yaml_config

Non re-esportiamo nulla qui: un import eager di `checkpoint` caricherebbe PyTorch e
romperebbe script leggeri (es. `python -m src.utils.monitor_metrics`) sul login node
senza torch installato.
"""
