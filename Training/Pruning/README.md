**[32]** What is the state of pruning
- Blalock, D., Ortiz, J. J. G., Frankle, J., & Guttag, J. (2020). What is the state of neural network pruning?. arXiv preprint arXiv:2003.03033. [Paper](https://arxiv.org/pdf/2003.03033.pdf)

# Code

## How to run the code?
```bash
main.py --epochs 10 --lr 0.1 --log-interval 100 --arch "my"
```
will run default

## Advanced run (IMPORTANT)
- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose

This post wass previously done at my [blog](https://www.subhadityamukherjee.me/2020/09/25/Pruning.html)

# Post
Today we will look at pruning and the different approaches followed.

Hello. It has been a long hiatus. I have been coding a lot but writing too less. Although I did end up writing a lot of documentation haha. I also changed my environment to Vim. (More on that later). Without further rants, lets get to todays topic.

> Note that most of this article is based on an excellent paper by Davis Blalock et al [WHAT IS THE STATE OF NEURAL NETWORK PRUNING?](https://arxiv.org/pdf/2003.03033.pdf)

[Cite] : Blalock, D., Ortiz, J. J. G., Frankle, J., & Guttag, J. (2020). What is the state of neural network pruning?. arXiv preprint arXiv:2003.03033.

# What is it

Pruning is something I have been interested in for a long time but somehow I could never get around to implementing it. It interested me for a lot of reasons. Mainly that of being able to reduce the size, cost and computational requirements of my models, all while maintaning the accuracy (sort of atleast). 

TL;DR Generally this comes about by removing parameters in some form or fashion. 

Rather than taking a mask, we can prune certain parts of the network by setting them to 0 or by dropping them if required. (aka weights and biases)

In most cases, the network is first trained for a while. Then pruned. Which reduces its accuracy and is thus trained again (fine tuning). This cycle is repeated until we get the results we require.

# Major types of pruning methods

There are many types of such methods but they have been categorized based on what they change from the original idea. Note that, in the end the main idea remains the same, just how it is done is varied.

## Structure
- Regarding structural choices, some authors choose to prune individual parameters which produces a sparse network (lots of 0s). This might not be very ideal for storing efficiently. 
- Some others consider methods where they group certain parameters and remove them as groups. This is more optimized.

## Scoring
- Like all networks, scoring becomes essential when we try to choose which parameter to get rid of. 
- Some authors suggest removing based on absolute values, others decide to prune based on the contributions of that parameter to the entire network. 
- Others remove based on a score given. 
- Some perform pruning locally, while others perform it globally across the network.

## Scheduling
- Some prune all the weights at once
- Others prune iteratively using loops or some other condition

## Fine tuning
- Some store weights before pruning and use that to continue training.
- Others somehow try to rewind to a previous state and reinitialize the network entirely

# Pruning heuristics + Code

Extending this from the previous topic and trying to code a bit.
 
**NOTE** These codes are TOY examples to understand the major ideas. For proper code refer to my repository here. [Link](https://github.com/SubhadityaMukherjee/pytorchTutorialRepo/tree/master/Pruning)

Let us get some random values in a nested array. In all cases, assume these are weights of a network. Each sub array represents a layer of the network let us say.

```jl
weights = [rand(10) for _ in 1:10]
```

We also consider a percentage of values/ number of values to drop as an input.

## Global Magnitude 

- Takes the lowest values in the entire network. Drops them.

Okay so we first flatten the nested array. Then we can easily find the n smallest values because it is a single long array now.
We can write a tiny function to identify if the value is greater than the a value and return 0 or the original otherwise.

```jl
function setval(val, cmpval, setter)
    if val<cmpval
        return setter
    else
        return val
    end
end
```

We can then sort this flattened list. After we do that, we run an element wise map function (over each layer) and pass each element to our previous function. This will allow us to set all the required elements to 0 (or our own value) if the current value is smaller than the value we chose. 
This will effectively drop the value.

```jl
"""
weights = array
n = lowest nth smallest value to prune below ( aka take the nth smallest value and prune below it ) (eg: 3)
"""
function global_mag_prune(weights, n = 3, setter = 0)
    temp = collect(Iterators.flatten(weights))
    sort!(temp)
    map.( x-> setval(x,temp[n], setter) ,  weights  )
    
end
```

We also write a function to determine how sparse the network has become. (Aka how many zeros).

```jl
return temp2, sum(x->x==0, collect(Iterators.flatten(temp2)) , dims=1)/(size(temp)[1]*size(temp)[1])*100
```

## Layerwise Magnitude

- Takes the lowest values per layer in the network and prunes.

Modifying the global layerwise and applying it per layer instead. 
To do this, we first make a copy of the weights. Then for every layer in the array, we find the least n values, take the nth value and set all the others to 0.
As an edge case, if the number of elements entered is greater than the total length of the layer, then the entire layer is set to 0. 

```jl
"""
weights = array
n = lowest nth smallest value to prune below ( aka take the nth smallest value and prune below it ) (eg: 3)
"""
function layer_mag_prune(weights, n = 3, setter = 0)
    backup = deepcopy(weights)
    for layer in 1:length(weights)
        sort!(backup[layer])
        if n>length(weights[layer])
            n = length(weights[layer])
        end
        backup[layer] = map.( x-> setval(x,weights[layer][n], setter) ,  weights[layer]  )
    end
    return backup, sum(x->x==0, collect(Iterators.flatten(backup)) , dims=1)/(size(weights)[1]*size(weights)[1])*100        
end
```

## Global Gradient Magnitude

- Identifies lowest absolute value (weight*gradient) in the whole network and removes them

For this we would need to compute the gradient of the weights so far. Well, how about I just use a random array as proof of concept so to speak.
So we basically take the previous code and change the inital part to being a product of the weights and the gradient. 

```jl
function global_grad_prune(weights, n = 3, setter = 0)
    temp = weights*rand(size(weights))
    temp = collect(Iterators.flatten(weights))
    sort!(temp)
    temp2 = map.( x-> setval(x,temp[n], setter) ,  weights  )
    return temp2, sum(x->x==0, collect(Iterators.flatten(temp2)) , dims=1)/(size(temp)[1]*size(temp)[1])*100
end
```

## Layerwise Gradient Magnitude

- Lowest absolute value per layer and removes them
Similarly, we modify this again to fit our needs

```jl
"""
weights = array
n = lowest nth smallest value to prune below ( aka take the nth smallest value and prune below it ) (eg: 3)
"""
function layer_grad_prune(weightsnew, n = 3, setter = 0)
    weights = deepcopy(weightsnew)*rand(size(weightsnew))
    backup = deepcopy(weights)
    for layer in 1:length(weights)
        sort!(backup[layer])
        if n>length(weights[layer])
            n = length(weights[layer])
        end
        backup[layer] = map.( x-> setval(x,weights[layer][n], setter) ,  weights[layer]  )
    end
        
    return backup, sum(x->x==0, collect(Iterators.flatten(backup)) , dims=1)/(size(weights)[1]*size(weights)[1])*100

end
```

## Random

- Each weight independantly considered and dropped with a fraction of network required

For this we first take the number of values to prune by identifying the total size of the weights and then multiplying it by the fraction of values to remove.
We then flatten the array, set the values we do not need to 0 and then reshape back to the original shape.

```jl
"""
frac = percentage of values to remove
"""
function random_prune(weights, frac = .3, setter = 0)
    num_prune = Int(round(frac*(size(weights)[1]*size(weights)[1])))
    @info num_prune
    backup = collect(Iterators.flatten(deepcopy(weights)))
    backup[rand(1:length(backup), num_prune)] .= 0
    @info size(weights)
    return reshape(backup, size(weights))
end
    
```

# What we can learn from papers (Thanks to the Blalock et al)

- Pruning methods cannot be decided by a single dataset but must be run using standardized tests over a variety of them to decide
- If a small compression is required, pruning might actually improve our accuracy in some cases
- Random pruning might not always be the best bet
- Global pruning is better
- Assigning different pruning percentages based on the layer type helps 
- Sparse models outperform dense ones generally 
- Pruned models can sometimes achieve higher accuracies than the initial architecture
- More effective with architectures that are poor in the first place
- Might be able to improve the time/space vs accuracy tradeoff

# That is awesome, but...

- You are sometimes off choosing a better and more efficient architecture sometimes
- It depends purely on what your objective is

# ~finis
That ends our little journey into pruning. Wow that took longer then expected. I should go sleep.
