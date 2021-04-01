using POMDPs: actions
using POMDPModelTools: ordered_states
using POMDPs: states, stateindex, convert_s
using LinearAlgebra
using POMDPs: actions, @gen, isterminal, discount, statetype, actiontype, simulate, states, reward
using POMDPSimulators: RolloutSimulator
using POMDPPolicies: FunctionPolicy
using D3Trees: inchrome
using StaticArrays: SA
using Statistics: mean


function ValueIteration(S,A,T,R,gamma,size,obj)
	V = rand(Float64,size) # this would be a good container to use for your value function
	V_p = zeros(Float64,size)
	A_s = length(A)
	iterations = 0

	while norm(V - V_p) > 10
	    V = copy(V_p)
	    actionValues = zeros(size,A_s)
	    for action in 1:A_s
	        actionValues[:,action] = R[A[action]] + gamma*T[A[action]][:, :]*V
		end
	    V_p = maximum(actionValues,dims=2)
	    @show iterations += 1
	end

	return V[:,1]
end
#For Phase 1 we need MCTS function, Phase 2 iterate over states w/ search
function MCTS(m,n,q,t,start,depth,iterations)
    count = 0
    c = 0.9
    s = start
    while count < iterations
        #Search
        act = search(m,s,n,q,t,depth)
        #[s,r] = @gen(m,s,act)

        if isterminal(m,s)
            break
        end
        count += 1
    end


end
function search(m,s,n,q,t,c,depth)
    count = 0
    while count < 7
        sim(depth, m, s,c,q,n,t)
        count += 1
    end
    return q,n,t
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
        return simulate(RolloutSimulator(max_steps=depth), m, FunctionPolicy(s->best_choice(m,s,7)), s)
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