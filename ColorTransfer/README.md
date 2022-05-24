## Color Transfer (Color Transfer)

### Terminologies

--k : number of mini-batches

--m : the size of mini-batches

--T : the number of steps

--cluster: K-means clustering to compress images

--palette: show the color palette

--source: Path to the source image

To run Color Transfer experiments in the paper:
```
python main.py  --m=100 --T=10000 --source images/s1.bmp --target images/t1.bmp --cluster

```
For more detailed settings, please check them in our papers.