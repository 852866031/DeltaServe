ps aux | grep 'ramya' | grep 'slora' | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux | grep 'ramya' | grep 'auto_benchmark.py ' | grep -v grep | awk '{print $2}' | xargs -r kill -9