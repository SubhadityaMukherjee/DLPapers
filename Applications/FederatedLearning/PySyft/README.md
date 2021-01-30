# Federated Learning using Pysyft

- [Link to main article](https://blog.openmined.org/upgrade-to-federated-learning-in-10-lines/)
- Demo Federated learning using two "simulated people"

## How to run?

### Default
```bash
python3 main.py --epochs 10 --lr 0.1 --log-interval 100 --arch "resnet18"
```
will run default 

- If you use "my" for the architecture, it will take it from Nets.py

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.
