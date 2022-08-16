file=open("outputData2.txt","r")
lines=file.readlines()
lines=[float(line.rstrip().split(", ")[7]) for line in lines]
file.close()
seedsum1=[]
seedsum2=[]
seedsum3=[]
seedsum4=[]
seedsum5=[]
for i in range(15):
    seedsum1.append(sum(lines[i*50:(i+1)*50])/50)
    seedsum2.append(sum(lines[750+i*50:750+(i+1)*50])/50)
    seedsum3.append(sum(lines[1500+i*50:1500+(i+1)*50])/50)
    seedsum4.append(sum(lines[2250+i*50:2250+(i+1)*50])/50)
    seedsum5.append(sum(lines[3000+i*50:3000+(i+1)*50])/50)

print(seedsum1)
print(seedsum2)
print(seedsum3)
print(seedsum4)
print(seedsum5)