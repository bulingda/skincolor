# skincolor
python extract skincolor  
  
method：  

1.k-means（final.py）  -三维（常规的k-means算法只对二维数据有效，因为不清楚多维空间中是否有把不为人知的结构所以欧氏距离不一定能用，另外一点是高维数据cluster后，无法判定他的准确性，只能投影到2维上来判定。如果维度太高，需要观察的就很多，可行性太低。解决方法：1.更好的了解问题；2.选择合适的方法降维），提取出来的颜色值整体要比方法2提取的颜色值低

2.subject_color(practice01.py) -函数原理暂时还没搞懂
