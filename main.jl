using POMDPs: actions
using POMDPModelTools: ordered_states
using POMDPs: states, stateindex, convert_s
using LinearAlgebra
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, reward, stateindex
using POMDPModelTools
using POMDPSimulators: RolloutSimulator
using POMDPPolicies: FunctionPolicy
using D3Trees: inchrome
using StaticArrays: SA
using Statistics
using StatsBase
using Plots

include("./gridworld.jl")


function ValueIteration(S,A,T,R,gamma,size,obj)
	V = rand(Float64,size) # this would be a good container to use for your value function
	V_p = zeros(Float64,size)
	A_s = length(A)
	iterations = 0

	while norm(V - V_p) > 0.001
	    V = copy(V_p)
	    actionValues = zeros(size,A_s)
	    for action in 1:A_s
	        actionValues[:,action] = R[A[action]] + gamma*T[A[action]][:, :]*V
		end
	    V_p = maximum(actionValues,dims=2)
	    #@show iterations += 1
	end

	return V[:,1]
end

function MCTS(m,n,q,t,start,depth,iterations)
    count = 0
	total_reward = 0
    c = 10
    s = start
    while count < iterations
        #Search
        act = search(m,s,n,q,t,c,depth)
        sp,r = @gen(:sp, :r)(m,s,act)
		total_reward += r

        if isterminal(m,sp)
            break
        end
		if r == -100.0
			break
		end
		s = sp
        count += 1
    end

	return total_reward
end
function search(m,s,n,q,t,c,depth)
    count = 0
    while count < 1000
        sim(depth, m, s,c,q,n,t)
        count += 1
    end
	val,index = findmax([q[s,:up],q[s,:down],q[s,:left],q[s,:right],q[s,:upLeft],q[s,:upRight],q[s,:downRight],q[s,:downLeft]])
	return actions(m)[index]
end

function best_choice(m,s,depth)
    rewards = []
    for a in actions(m)
        p = FunctionPolicy(s->a)
        rt = simulate(RolloutSimulator(max_steps=depth), m, p, s)
        push!(rewards,rt)
    end
    val,ind = findmax(rewards)
    return actions(m)[ind]
end

function sim(depth, m, s, c,q,n,t)
    if depth <= 0 #at max depth
        return 0.0
    end
    if isterminal(m,s)
        return r
    end
    if !haskey(n,(s,actions(m)[1])) #add all action nodes to dict
        for a in actions(m)
            n[s,a] = 0
            q[s,a] = 0
            #t[s,a,@gen(:sp)(m,s,a)] = 0
        end
        return simulate(RolloutSimulator(max_steps=depth), m, FunctionPolicy(s->best_choice(m,s,50)), s)
    end
    #Find UCB recommended action
    acts = []
    sum = n[s,:up]+ n[s,:down] + n[s,:left] + n[s,:right]
    for a in actions(m)
        if n[s,a] == 0
            push!(acts,10000)
        else
            push!(acts,q[s,a] + c*sqrt(log(sum)/n[s,a]))
        end
    end
    val,index = findmax(acts)
    act = actions(m)[index]
    #Take UCB action
    sp,r = @gen(:sp, :r)(m,s,act)
    t[(s, act, sp)] = get(t, (s, act, sp), 0) + 1

    if isterminal(m,sp)
        #q[s,act] += r #rest of bellman is zero for terminal
        return r #no need to simulate from here
    end

    n[s,act] += 1

    #Not a new node?
    q_value = r + m.discount*sim(depth-1, m, sp, c,q,n,t)
    q[s,act] += (q_value - q[s,act])/n[s,act]
    return q_value
end

function SQ(MCTS_samples, VI_samples, rH, rL)
	p_samples = countmap(MCTS_samples)
	q_samples = countmap(VI_samples)
	all_samples = [MCTS_samples VI_samples]
	#Make pdfs
	h = 0
	p = []
	q = []
	for x in 1:length(p_samples)
		push!(p,p_samples[x])
	end
	for x in 1:length(q_samples)
		push!(q,q_samples[x])
	end

	for sample in all_samples
		if !haskey(p_samples,sample)
			p_i = 0
		else
			p_i = p[sample]
		end
		if !haskey(q_samples,sample)
			q_i = 0
		else
			q_i = q[sample]
		end
		h += (sqrt(p_i) - sqrt(q_i))^2
	end
	h = 1/sqrt(2)*h
	f = (mode(MCTS_samples)-mode(VI_samples))/(rH-rL)
	q = sign(mean(MCTS_samples) - mean(VI_samples))*f^0.1*sqrt(h)
	SQ = 2/(1+exp(-q/5))

	return SQ
end
function getAction(x, V, R, T, A)
	y = []
	for a in 1:length(A)
		push!(y,R[A[a]][stateindex(m,x)] + sum(V[stateindex(m,x)]*T[A[a]][stateindex(m,x),:]))
	end
	val,ind = findmax(y)
	return actions(m)[ind]
end

function MCSamples(m,start,V,R,T,A)
	s = start
	r = 0
	rew = 0
	while isterminal(m,s) == false
		act = getAction(s,V,R,T,A)
		@show act
		sp,rew = @gen(:sp, :r)(m,s,act)
		r += rew
		if rew == -100.0
			break
		end
		s = sp
	end
	return r
end
#Phase 1: Simlulate VI, MCTS
N_s = 100
m = SimpleGridWorld()
T = transition_matrices(m)
R = reward_vectors(m)
all_states = states(m)
all_actions = actions(m)

V = ValueIteration(all_states,all_actions,T,R,m.discount,401,m)
#Need to simulate with V now

S = statetype(m)
A = actiontype(m)
n = Dict{Tuple{S, A}, Int}() #number of times node has been tried
q = Dict{Tuple{S, A}, Float64}() #Q values
t = Dict{Tuple{S, A, S}, Int}() #times transition was generated

start = [11,19]
VIreward = []
MCTSreward = []
for i = 1:N_s
	@show i
	r1 = MCSamples(m,start,V,R,T,actions(m))
	#r2 = MCTS(m,n,q,t,start,100,100)
	push!(VIreward,r1)
	#push!(MCTSreward,r2)
end
histogram(VIreward)
