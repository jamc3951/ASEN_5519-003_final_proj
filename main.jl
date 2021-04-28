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


function ValueIteration(S,A,T,R,gamma,size,obj,iter)
	V = rand(Float64,size) # this would be a good container to use for your value function
	V_p = zeros(Float64,size)
	A_s = length(A)
	iterations = 0

	#while norm(V - V_p) > 0.001
	while iterations < iter
	    V = copy(V_p)
	    actionValues = zeros(size,A_s)
	    for action in 1:A_s
	        actionValues[:,action] = R[A[action]] + gamma*T[A[action]][:, :]*V
		end
	    V_p = maximum(actionValues,dims=2)
	    iterations += 1
	end

	return V[:,1]
end

function MCTS(m,n,q,t,start,depth,iterations,costmap,goal,iter)
    count = 0
	total_reward = 0
    c = 1
    s = start
    while count < iterations
        #Search
        act = search(m,s,n,q,t,c,depth,costmap,goal,iter)
        sp,r = @gen(:sp, :r)(m,s,act)
		total_reward += r

		if sp == goal
			total_reward+= 200.0
			break
		end

        if isterminal(m,sp)
            break
        end
		if r == -100.0
			break
		end
		#@show s,sp, act
		s = sp
        count += 1
    end

	return total_reward
end
function search(m,s,n,q,t,c,depth,costmap,goal,iter)
    count = 0
    while count < iter
        sim(depth, m, s,c,q,n,t,costmap,goal)
        count += 1
    end
	val,index = findmax([q[s,:up],q[s,:upRight],q[s,:upLeft],q[s,:down],q[s,:downLeft],q[s,:downRight],q[s,:left],q[s,:right]])

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

function sim(depth, m, s, c,q,n,t,costmap,goal)
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
        return estimateValue(costmap,s,goal)
    end
    #Find UCB recommended action
    acts = []
    num = n[s,:up]+ n[s,:down] + n[s,:left] + n[s,:right]+ n[s,:upRight]+ n[s,:upLeft] + n[s,:downRight] + n[s,:downLeft]
    for a in actions(m)
        if n[s,a] == 0
            push!(acts,10000)
        else
            push!(acts,q[s,a] + c*sqrt(log(num)/n[s,a]))
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
	if r == -100.0
		return r
	end
    n[s,act] += 1

    #Not a new node?
    q_value = r + m.discount*sim(depth-1, m, sp, c,q,n,t,costmap,goal)
    q[s,act] += (q_value - q[s,act])/n[s,act]
    return q_value
end
function estimateValue(costmap,s,goal)
	d2g = costmap[goal[1],goal[2],s[1],s[2]]

	if d2g == Inf || d2g == -Inf
		return -500.0
	end
	if d2g == 0.0
		return 2000.0
	else
		return 200.0 - abs(d2g*-2.0)
	end
end
function floydWarshall(grid,sizeX,sizeY)
	dist = ones(sizeX,sizeY,sizeX, sizeY)*Inf

	for i in 1:sizeX
		for j in 1:sizeY
			dist[i,j,i,j] = 0.0
		end
	end

	for i in 1:sizeX
		for j in 1:sizeY
			if (i > 1) && (grid[i-1,j] == 0.0)
				dist[i,j,i-1,j] = 1.0
			end
			if (i < sizeX - 1) && (grid[i+1,j] == 0.0)
				dist[i,j,i+1,j] = 1.0
			end
			if (j>1) && (grid[i,j-1] == 0.0)
				dist[i,j,i,j-1] = 1.0
			end
			if (j < sizeY -1) && (grid[i,j+1] == 0.0)
				dist[i,j,i,j+1] = 1.0
			end
		end
	end
	for kx in 1:sizeX
		for ky in 1:sizeY
			for ix in 1:sizeX
				for iy in 1:sizeY
					for jx in 1:sizeX
						for jy in 1:sizeY
							if dist[ix,iy,jx,jy] > (dist[ix,iy,kx,ky] + dist[kx,ky,jx,jy])
								dist[ix,iy,jx,jy] = dist[ix,iy,kx,ky] + dist[kx,ky,jx,jy]
							end
						end
					end
				end
			end
		end
	end
	return dist




end

function SQ(MCTS_samples, VI_samples, rH, rL, alpha)
	p_samples = countmap(MCTS_samples)
	q_samples = countmap(VI_samples)
	all_samples = push!(MCTS_samples,VI_samples)
	#Make pdfs
	h = 0
	p = []
	q = []
	for x in keys(p_samples)
		push!(p,p_samples[x]/length(MCTS_samples))
	end

	for x in keys(q_samples)
		push!(q,q_samples[x]/length(VI_samples))
	end

	for sample in 1:length(unique(all_samples))
		q_i = 0
		p_i = 0
		if !haskey(p_samples,all_samples[sample]) && sample < length(unique(MCTS_samples))
			p_i = 0
		elseif sample < length(unique(MCTS_samples))
			p_i = p[sample]
		end
		if !haskey(q_samples,all_samples[sample]) && sample > length(unique(MCTS_samples))
			q_i = 0
		elseif sample > length(unique(MCTS_samples))
			q_i = q[sample]
		end
		h += (sqrt(p_i) - sqrt(q_i))^2
	end
	h = 1/sqrt(2)*h
	#@show h
	f = (mode(MCTS_samples[1])-mode(VI_samples[1]))/(rH-rL)
	#@show f
	q = sign(mean(MCTS_samples[1]) - mean(VI_samples[1]))*abs(f)^alpha*sqrt(h)
	#@show sign(mean(MCTS_samples[1]) - mean(VI_samples[1]))
	SQ = 2/(1+exp(-q/5))

	return SQ
end
function getAction(x, V, R, T, A)
	y = zeros(401,length(A))
	for action in 1:length(A)
		y[:,action] = R[A[action]] + T[A[action]][:, :]*V
	end
	val,ind = findmax(y[stateindex(m,x),:])
	return actions(m)[ind]
end

function MCSamples(m,start,V,R,T,A)
	s = start
	r = 0
	rew = 0
	while isterminal(m,s) == false
		act = getAction(s,V,R,T,A)
		sp,rew = @gen(:sp, :r)(m,s,act)
		r += rew
		if rew == -100.0 || rew == 200.0 || r < -1000.0
			break
		end
		s = sp
	end
	return r
end

function SQ2(map_,V,R,T,A,sizeX,sizeY,costmap,goal,c,depth,n,q,t,iter)
	#Assume same dist. to start
	SQ = []
	count = 0.0
	better = 0.0
	#At each state, what does MCTS, VI think you should do
	#Which is objectively better
	for i in 1:sizeX
		for j in 1:sizeY
			if map_[i,j] != 1.0 && goal != [i,j]
				VI_act = getAction([i,j],V,R,T,A)
				MCTS_act = search(m,[i,j],n,q,t,c,depth,costmap,goal,iter)

				if VI_act != MCTS_act
					count += 1.0
					sp_VI,rVI = @gen(:sp, :r)(m,[i,j],VI_act)
					sp_MCTS,rMCTS = @gen(:sp, :r)(m,[i,j],MCTS_act)
					d2g1 = costmap[goal[1],goal[2],sp_VI[1],sp_VI[2]]
					d2g2 = costmap[goal[1],goal[2],sp_MCTS[1],sp_MCTS[2]]
					#Which is better?
					VI_val = R[VI_act][stateindex(m,[i,j])] + (m.discount^d2g1)*200.0
					MCTS_val = R[MCTS_act][stateindex(m,[i,j])] + (m.discount^d2g2)*200.0

					if MCTS_val >= VI_val
						better += 1.0
					end
				end
			end
		end
	end

	return (better/count)*2
end

function benchmarkSQ(map_,all_states,all_actions,T,R,discount,size,m,start,costmap,goal)
	iteration_list = 2:5:30
	search_list = 1:10:100
	SQ1 = zeros(length(iteration_list),length(search_list))
	SQtwo = zeros(length(iteration_list),length(search_list))
	SQ3 = zeros(length(iteration_list),length(search_list))
	for i in 1:length(iteration_list)
		V = ValueIteration(all_states,all_actions,T,R,discount,size,m,iteration_list[i])
		for j in 1:length(search_list)
			VIreward = []
			MCTSreward = []
			n = Dict{Tuple{S, A}, Int}() #number of times node has been tried
			q = Dict{Tuple{S, A}, Float64}() #Q values
			t = Dict{Tuple{S, A, S}, Int}() #times transition was generated
			#Phase 2

			SQtwo[i,j] = SQ2(map_,V,R,T,actions(m),20,20,costmap,goal,1,100,n,q,t,search_list[j])
			#Phase 3
			for k = 1:20
				@show k
				r1 = MCSamples(m,start,V,R,T,actions(m))
				r2 = MCTS(m,n,q,t,start,100,80,costmap,goal,search_list[j])
				push!(VIreward,r1)
				push!(MCTSreward,r2)
			end
			SQ3[i,j] = SQ(copy(MCTSreward),copy(VIreward),200.0,-70.0,7)
			#Phase 1
			SQ1[i,j] = SQ(copy(MCTSreward),copy(VIreward),200.0,-260.0,0.5)
			@show (iteration_list[i],search_list[j],SQ1[i,j],SQtwo[i,j],SQ3[i,j])
		end
	end
	return SQ1,SQtwo,SQ3
end

#Phase 1: Simlulate VI, MCTS
N_s = 100
m = SimpleGridWorld()
T = transition_matrices(m)
R = reward_vectors(m)
all_states = states(m)
all_actions = actions(m)

V = ValueIteration(all_states,all_actions,T,R,m.discount,401,m,30)
costmap = floydWarshall(map_,20,20)
#Need to simulate with V now

S = statetype(m)
A = actiontype(m)
n = Dict{Tuple{S, A}, Int}() #number of times node has been tried
q = Dict{Tuple{S, A}, Float64}() #Q values
t = Dict{Tuple{S, A, S}, Int}() #times transition was generated

start = [13,19]
goal = [2,9]
VIreward = []
MCTSreward = []
for i = 1:N_s
	@show i
	r1 = MCSamples(m,start,V,R,T,actions(m))
	r2 = MCTS(m,n,q,t,start,100,80,costmap,goal,1)
	push!(VIreward,r1)
	push!(MCTSreward,r2)
end
display(histogram(VIreward,title="Histogram of VI Rollouts", xlab = "Cumulative Reward", ylab = "Count out of 100",bins=100))

SQval = SQ(copy(MCTSreward),copy(VIreward),200.0,-70.0,7)
SQnew = SQ2(map_,V,R,T,actions(m),20,20,costmap,goal,1,100,n,q,t,1)
@show SQval
@show SQnew


p1,p2,p3 = benchmarkSQ(map_,all_states,all_actions,T,R,m.discount,401,m,start,costmap,goal)
#display(contourf(1:6,1:10,(x,y) -> p3[x,y],xaxis = ("VI Iterations [2,30)"),yaxis= ("MCTS Search Time [1,100]"),zlabel = ("SQ"),title = "Phase 3 SQ Variability"))
