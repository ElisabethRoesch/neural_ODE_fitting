using Distances

a = fill(1.5,(8,8))
b = fill(0.1,(8,8))

dist = hellinger
r = dist(Array(a[1,:]), Array(b[1,:]))
reshape(a[1,:],1,length(a[1,:]))
Array(a[1,:])
r = colwise(dist, a, b)

pairwise(dist,a,b,dim =2)

for i in a[1,:]
