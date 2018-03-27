binwidth=5
bin(x,width)=width*floor(x/width)

plot 'grads.txt' using (bin($1,binwidth)):(1.0) smooth freq with boxes