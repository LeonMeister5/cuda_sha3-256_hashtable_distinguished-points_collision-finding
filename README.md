# How to use
This is a collision finding project. With RTX4060 in my laptop, I'm able to find 80 bit collision in about 1 hour. The project workds completely in GPU. 

Sha3.cuh is sha3. ht.cuh is hashtable.

With dp.cu, you are ablt to find collision. You should reset prefix, n_bits and m_bits for your convenience. Then in the commamd line output, there are two seeds.

Then you should manually change rec.cu's main function, input the seeds you got. Then the command line output will contain the collision. 

Then you can test the result with test.ipynb. 

# Cuda thread protection
ht.cuh only compile on windows. 

atomicCAS only protect 64 bit. If you work on linux and want to change hashtable, keep extra eye on race condition. 

Insert is negligible compared with search, so you can protect insert and don't project search.

# Algorithm
Start from random n_bit seed, go step by step and find a m_bit distinguished points. 

In steps, only use first n_bit seed. This allows the project to run in 2^n space, instead of 2^256 space. 

Hashtable key-value pair is (dp, seed). Finding dp collision returns seed1 and seed2. Then we reconstruct seed1->dp and seed2->dp, storing every point in the first chain in hash table, and in chain2, go 1 step, search in hash table. Then we find where the 2 chains merge into one. 
