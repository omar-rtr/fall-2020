# Ruiz Omar, auditing the course
function q1()
@eval using Random
@eval using Distributions
@eval using LinearAlgebra
# a) We set the seed 1234
Random.seed!(1234)
# i. A 10x7 matrix  with random numbers U[-5,10]
A=rand(Uniform(-5,10),10,7)
# ii. B 10x7 matrix  with random numbers N(-2,15)
B=rand(Normal(-2.0,15.0),10,7)
# iii. C 5x7 matrix with first five rows and columns of A and first five rows and two last columns of B 
C=[A[1:5,1:5] B[1:5,6:7]]
# iv. D a matrix with values of A but if they are positive, convert them to zeros.
D=A
D[D.>0].=0
D
# b) List the number of elements of A
prod(size(A))
# c) List unique elements of D
prod(size(unique(D)))
# d) Vectorize B and call it E with reshape
E=reshape(B,70,1)
# Alternative solution, not easier than wih reshape
E=[B[1:10,1]; B[1:10,2]; B[1:10,3] ; B[1:10,4]; B[1:10,5]; B[1:10,6]; B[1:10,7]]
# e) F, an array with A and B in the first and second columnsof the third dim, respectively
F=reshape([A B], (10,7,2))
# f) Reshape F so it is now a 2x7x10 array
F=permutedims(F, [3,2,1])
# g) G computes the Kronecker product of B and C
G=kron(B,C)
# When trying the same with C and F a problem occurs due to the fact that F is not a matrix, but an 
# array containing 10 different so the Kronecker product is not computed.
# h) Save A,B,C,D,E,F and G as matrixpractice.jld
@eval using JLD
save("/matrixpractice.jld","A",A,"B",B,"C",C,"D", D,"E",E,"F",F,"G",G)
# i) Save matrices A-D as firstmatrix.jld
save("/firstmatrix.jld","A",A,"B",B,"C",C,"D", D)
# j) Export C as a csv file
@eval using CSV
@eval using DataFrames
C_1=DataFrame(C)
CSV.write("Cmatrix.csv", C_1)
# k) Export D as a tab-delimited file, called Dmatrix.dat
@eval using DelimitedFiles
writedlm("Dmatrix.dat", D)
return A, B, C, D
end
A, B, C, D = q1()

# 2 Loops and comprehensions
function q2(A,B,C)
    # a) Write a loop or a comprehension for the element-by-element product of A and B.
    AB=map((A,B) -> A.*B, A, B)
    # a) Create a matrix called AB2 that accomplishes the same without a comprehension or a loop
    AB2=A.*B
    # b) Write a loop that creates a column vector Cprime which contains the elements of C between -5 and 5 
    Cprime=Float64[]
    for i in C
     if 5.0>=i>=-5.0 
        append!(Cprime, i)
     end
    end
    # b) Create a vector Cprime2 that does the same without a loop
    Cprime2=filter(x -> -5.0<=x<=5.0,reshape(C,35,1))
    # c) Crate a 3-dim array called X of dimension NxKxT 15169x6x5
    n=15169
    T=5
    for i in 1:T
        local onesx=ones(n)
        local DM=rand(n)
        for p in eachindex(DM)
            if DM[p] > 0.75*(6-i)/5 
                DM[p]=0
            else DM[p]=1
            end
        end
        local NV=rand(Normal(15.0+i-1,5.0*(i-1)),n,1)
        local NEV=rand(Normal(π*(6-i)/3,1/ℯ),n,1)
    # A discrete normal distribution is the limiting case when a binomial distribution has N increasing to infinity
    # For the case in which the mean is 12 and its standard deviation is 2.19^3 we can use the approximation of 
    # a N->∞ binomial which mean is μ=Np and sd is σ^2=Np(1-p). So if we take as given mean at 12 and solve for p and then substitute in
    # σ^2=Np(1-p) and solve for N, we get that N=96.2219 and p=0.12471173 and q=1-p=0.875288266
    # Using these parameters (rounded) to approximate a discrete normal distribution through a binomial distribution:
        local DN=rand(Binomial(96,0.125),n,1)
        local BN=rand(Binomial(20,0.5),n,1)
        if i==1 
        global    H=[onesx DM NV NEV DN BN]
        else 
        global    H=[H;[onesx DM NV NEV DN BN]]
        end
    end    
    X=reshape(H, (n,6,5))
    
    # d) β KxT matrix with evolution through time is as given. Use comprehensions.
    β=[[1+(i-1)*0.25 for i in 1:5] [log(i) for i in 1:5] [-√i for i in 1:5] [ℯ^i - ℯ^(i+1)  for i in 1:5] [i  for i in 1:5] [i/3  for i in 1:5]]
    
    # e) Y is a NxT defined by Y_t=X_tB_t+e_t where e_t is N(0,σ=0.36)
    for i in 1:5
        if i==1 
        global Y=X[:,:,1]*β[1,:]+rand(Normal(0,0.36),n,1)
        else
        global Y=[Y X[:,:,i]*β[i,:]+rand(Normal(0,0.36),n,1) ]
        end
    end
    end
    
q2(A,B,C)

function q3()
  # a) Clear workspace and read file, 
   @eval using DataFrames
   @eval using CSV
   @eval using HTTP
   # I correct for misidentified data types
   nls= CSV.read(download("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"), missingstrings=["NA", "na", "n/a", "nothing","missing"],types=[Int,Int,String,Int,Int,Int,Int,Int,Int,Int,String,String,Int,Float64,Int,Float64,Float64], silencewarnings=true)
   DataFrame(nls)
   describe(nls)
   save("/nls88.jld","nls88",nls)
   # b) Percentage of sample never been married - no information is provided about marriage status
   mean(nls.never_married)*100
   # b) To compute percentage of college graduates, we use the variable collgrad
   mean(nls.collgrad)*100
   # c) What percentage of sample is in each race category?
   c=nrow(nls)
   count=by(nls,:race,nrow)
   count.percent=count.nrow/c*100
   count
   # d) summarystats is a matrix with mean, median, sd, min, max, unique, interquantile range of nls
   # I define the dataframe to work with
   summarystats=describe(nls, :mean, :median, :std, :min, :max, :nunique, :q75, :q25 )
   DataFrame(summarystats)
   # Since interquantile range is not provided directly by describe, we compute it skipping those non-numeric
   # variables 
   c=nrow(summarystats)
   summarystats[:intqtile]=zeros(c)
   for i=1:c
    if summarystats[i,:q75]==nothing 
    else summarystats[i,:intqtile]=summarystats[i,:q75]-summarystats[i,:q25] 
    end
   end 
  # e) cross-tabulation of industry and occupation
  count=by(nls,[:occupation, :industry],nrow) 
  # count shows the cross-tabulation with levels of industrya in rows and levels of occupation as column names
  count=unstack(count, :occupation, :industry, :nrow)
  # f) Mean wage over industry and occupation categories
  nlsn=select!(nls,:industry, :occupation, :wage)
  # Mean wage over industry categories
  nlsna=groupby(nlsn,:industry)
  combine(nlsna, :wage => mean)
  # Mean wage over occupation categories
  nlsna=groupby(nlsn,:occupation)
  combine(nlsna, :wage => mean)
  # Mean wage over occupation-industry categories
  nlsna=groupby(nlsn,[:occupation,:industry])
  combine(nlsna, :wage => mean)
end
q3()

function q4()
    # a) Load firstmatrix.jld
   @eval using JLD
   @eval using LinearAlgebra
    fm=load("/firstmatrix.jld")
    # b) write a function called matrixops 
    function matrixops(A,B)
    # c) This function computes and prints element by element product of both matrices AB, product of A'B (ApB) and the sum of each entry in A+B
        sa=size(A)
        sb=size(B)
     # e) Error if not the same size
        if sa!=sb
            println("inputs must have the same size")
        else
        AB=map((A,B) -> A.*B, A, B)
        # product A'B 
        ApB=transpose(A)*B
        # sum of all elements of A+B
        sAB=sum(A+B)
        return AB, ApB, sAB
        end
    end
    # d) Evaluate using A and B 
    matrixops(A,B)
    # f) Evaluate using C and D
    matrixops(C,D)
    # g) Evaluate matrixops with ttl_exp and wage 
    @eval using CSV
    nls= CSV.read(download("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"), missingstrings=["NA", "na", "n/a", "nothing","missing"],types=[Int,Int,String,Int,Int,Int,Int,Int,Int,Int,String,String,Int,Float64,Int,Float64,Float64], silencewarnings=true)
    J=convert(Array,nls.ttl_exp)
    L=convert(Array,nls.wage)
    matrixops(J,L)
end
q4()
