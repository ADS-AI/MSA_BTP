#MSA Toolbox

- For running, choose an one of one the config from 'cfgs' folder.
- Each config has format 'config_activelearningname.yaml' where 'activelearningname' is the name of the active learning strategy.
- IMPORTANT: Set the appropriate parameters in the config file.
- To run the code, use the following command:
```
from msa_toolbox import main
path = #path-to-the-config-file
main.app(path)
```