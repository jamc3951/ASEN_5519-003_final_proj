using StatsFuns

function goa(data, outcome_partitions, z_star)
    # get # of bins
    num_bins = length(outcome_partitions)-1

    if z_star < 1 || z_star > num_bins
    	return
	end

    # ranked z-domain
    z_domain = []
    for i = 1:num_bins
        z_i = []
		for val in data
			if val <= outcome_partitions[i+1] && val > outcome_partitions[i]
				push!(z_i,val)
			end
		end
        push!(z_domain,z_i)
	end
    # estimate the probabilities in each bin
    p_z = []
    for i = 1:length(z_domain)
        current_bin = z_domain[i]
        push!(p_z,float(length(current_bin)) / float(length(data)))
	end

    # compute UPM/LPM
    d_lpm = 0
    d_upm = 0

    for i = 1:length(z_domain)
        if i < z_star
            d_lpm += (z_star - i) * p_z[i]
        elseif i >= z_star
            d_upm += (i - z_star + 1) * p_z[i]
		end
	end
    # compute GOA
    if d_lpm == 0
        outcome_assessment = 1
    elseif d_upm == 0
        outcome_assessment = -1
    else
        outcome_assessment = 2 / (1 + exp(-log(d_upm / d_lpm))) - 1
	end
    return outcome_assessment
end

function MCTS2(m,Tm,n,q,t,start,depth,iterations,costmap,goal,iter)
    count = 0
	total_reward = 0
	tp = 1.0
	outcome = 0.0
    c = 1
    s = start
	acts = []
	states = []
    while count < iterations
        #Search
        act = search(m,s,n,q,t,c,depth,costmap,goal,iter)
        sp,r = @gen(:sp, :r)(m,s,act)
		tp = tp* Tm[act][stateindex(m,s),stateindex(m,sp)]
		total_reward += r
		push!(acts,act)
		push!(states,s)
		if sp == goal
			outcome = 1.0
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

	return acts, outcome, tp, states
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
function goa_online(m,n,q,t,start,depth_list,costmap,goal,N)
	s = start
	confidence = []
	r = 0
	rew = 0
	n = Dict{Tuple{S, A}, Int}() #number of times node has been tried
	q = Dict{Tuple{S, A}, Float64}() #Q values
	t = Dict{Tuple{S, A, S}, Int}() #times transition was generated
	while isterminal(m,s) == false
		outcomes = []
		for i = 1:N
			acts, o = MCTS2(m,copy(n),copy(q),copy(t),s,depth_list,80,costmap,goal,100)
			push!(outcomes,o)
		end
		push!(confidence,goa(outcomes,[-0.5,0.5,1],2))
		act = search(m,s,n,q,t,1,depth_list,costmap,goal,100)
		sp,rew = @gen(:sp, :r)(m,s,act)
		@show sp
		r += rew
		if rew == -100.0 || rew == 200.0 || r < -1000.0
			break
		end
		s = sp
	end
	return confidence
end
function SIS2(m,Tm,N_s,C_s,s,s_old,goal,n,q,t,depth_list,costmap, hist_sample,w_p)
	current_kp = []
	adversary_kp = []
	outcomes_kp = [0.0,0.0]
	#alpha = 0.75;
	all_weights = []
	transitions = []
	outcomes = []
	traj_actions = []
	char_weights = []
	cs_states = []
	#Characteristic Samples

	for j = 1:C_s
	    #Sample x_k+1
	    action, outcome, tp, states = MCTS2(m,Tm,copy(n),copy(q),copy(t),s,depth_list,80,costmap,goal,100)
		#Update action map
		#=for i = 1:length(states)
			if !haskey(a2,(states[i],action[i]))
				for n in actions(m)
					a[states[i],n] = 0
				end
			else
				a2[states[i],action[i]] += 1
			end
		end=#


	    push!(outcomes,outcome)
	    push!(traj_actions,[s,outcome,action])
		push!(transitions,tp)
	    #char_weights(j)=mvnpdf([current_kp(j,:),adversary_kp(j,:)],ykp,abs([current_kp(j,:),adversary_kp(j,:)]- ykp).*eye(4) + eye(4));
	    push!(char_weights,1.0)
		push!(cs_states,states)

	    if outcomes[j] == 1
	        outcomes_kp[2] = outcomes_kp[2] + char_weights[j]
	    else
	        outcomes_kp[1] = outcomes_kp[1] + char_weights[j]
	    end

	    push!(all_weights,1)
	end

	char_weights=char_weights./sum(char_weights);
	wkp=outcomes_kp
	outcomes_fkp = [0.0,0.0]
	c = 0
	accept = 0
	@show wkp
	if !isempty(hist_sample)
		for acts in hist_sample
			c+=1
			follower_weights = [0,0]

			for i = 1:500
				outc = 0
				rew = 0.0
				st = [acts[1][1],acts[1][2]]
				iterations = 0
				for ind_acts in acts[3]
					sp,rew = @gen(:sp, :r)(m,st,ind_acts)

					if rew == 200.0
						outc = 1
						break
					end
					if rew == -100.0
						outc = 0
						break
					end
					st = sp
					#buuucket
					if abs(sum(st - goal)) < 4
						outc = 1
						break
					end
				end
				if outc == 1
					follower_weights[2] = follower_weights[2] + 1
				else
					follower_weights[1] = follower_weights[1] + 1
				end
			end
			#@show follower_weights
			follower_weights=follower_weights./sum(follower_weights);
			if reject(wkp./sum(wkp),follower_weights,0.1) == 1
				accept += 1
				if acts[2] == 1
					outcomes_fkp[2]+=1
				else
					outcomes_fkp[1]+=1
				end
			end
		end
		@show outcomes_fkp
		if sum(outcomes_fkp) > 20
			outcomes_kp =  (outcomes_fkp./sum(outcomes_fkp))
		else
			accept = 0
			outcomes_kp = wkp
		end
	end
	@show sum(outcomes_fkp)


	w_k= outcomes_kp./sum(outcomes_kp)
	return w_k, traj_actions, accept
end

function reject(observed,simulated,ep)
	x = observed-simulated
	#@show (x[1]),(x[2])
	if abs(x[1]) < ep
		return 1
	end
	return 0
end

function traj_distance(C_s,cs_traj,eval_traj,epsilon)
	for i = 1:C_s
		c = 0
		current_comp = cs_traj[i]
		for j = 1:min(length(current_comp),length(eval_traj))
			if current_comp[j] == eval_traj[j]
				@show current_comp[j]
				c += 1.0
			end
		end
		@show c/min(length(current_comp),length(eval_traj))
		if c/min(length(current_comp),length(eval_traj)) >= epsilon
			return 1
		end
	end
	return 1
end

function BS(m,Tm,n,q,t,s,depth_list,costmap,goal, o_true, o_bad, abc_weights, N)
	bs_true = 0
	bs_bad = 0
	bs_abc = 0

	ps_true = sum(o_true)/length(o_true)
	ps_bad = sum(o_bad)/length(o_bad)
	for i = 1:N
		acts, o, tp, states = MCTS2(m,Tm,copy(n),copy(q),copy(t),s,depth_list,80,costmap,goal,400)

		bs_abc += (abc_weights[2] - o)^2
		bs_true += (ps_true - o)^2
		bs_bad += (ps_bad - o)^2
	end
	bs_abc = bs_abc ./N
	bs_true = bs_true ./N
	bs_bad = bs_bad ./N

	return bs_abc,bs_true,bs_bad
end

function goa_pf(m,Tm,start,depth_list,costmap,goal,N_s,C_s)
	s = start
	confidence = []
	confidence_standard = []
	confidence_truth = []
	r = 0
	iterations = 15
	count = 0
	rew = 0
	n = Dict{Tuple{S, A}, Int}() #number of times node has been tried
	q = Dict{Tuple{S, A}, Float64}() #Q values
	t = Dict{Tuple{S, A, S}, Int}() #times transition was generated
	char_k = []
	s_old = []
	accepts_list = []
	hist_list = []
	bs_abc = []
	bs_true = []
	bs_std = []
	for i = 1:3*C_s
		acts,o, tp, states = MCTS2(m,Tm,copy(n),copy(q),copy(t),s,depth_list,80,costmap,goal,400)
		push!(char_k,[s,o,acts])
	end
	push!(hist_list,length(char_k))
	w_k = [0.5,0.5]
	while count < iterations
		outcomes = []
		outcomes_true = []

		w_k, ch_k, accepts = SIS2(m,Tm,N_s,C_s,s,s_old,goal,copy(n),copy(q),copy(t),depth_list,costmap, char_k, w_k)
		append!(char_k,ch_k)
		push!(hist_list,length(char_k))
		append!(accepts_list, accepts)
		@show w_k

		for i = 1:12
			acts, o, tp, states = MCTS2(m,Tm,copy(n),copy(q),copy(t),s,depth_list,80,costmap,goal,400)
			push!(outcomes,o)
		end

		for i = 1:50
			acts, otrue , tp, states = MCTS2(m,Tm,copy(n),copy(q),copy(t),s,depth_list,80,costmap,goal,400)
			push!(outcomes_true,otrue)
		end
		push!(confidence_standard,goa(outcomes,[-0.5,0.5,1],2))
		push!(confidence_truth,goa(outcomes_true,[-0.5,0.5,1],2))
		push!(confidence,goa(float(sample([0.0,1.0],Weights(w_k),N_s)),[-0.5,0.5,1.0],2))

		bs1,bs2,bs3 = BS(m,Tm,copy(n),copy(q),copy(t),s,depth_list,costmap,goal, outcomes_true, outcomes, w_k, 10)
		push!(bs_abc,bs1)
		push!(bs_true,bs2)
		push!(bs_std,bs3)

		act = search(m,s,n,q,t,1,depth_list,costmap,goal,100)
		sp,rew = @gen(:sp, :r)(m,s,act)

		r += rew
		if rew == -100.0 || rew == 200.0 || r < -1000.0
			@show rew
			break
		end
		append!(s_old,s)
		s = sp
		count += 1
	end
	return confidence, confidence_standard, confidence_truth, hist_list, accepts_list, bs_abc,bs_true, bs_std
end

function OA(samples, costmap, start, goal)
	p_samples = countmap(samples)
	d2g = costmap[goal[1],goal[2],start[1],start[2]]
	R_inf = 0#200.0 - abs((d2g + 15.0)*-2.0)
	#@show R_inf
	L_samples = []
	U_samples = []
	s = copy(samples)
	samples = unique(samples)
	for sample in samples
		if sample < R_inf
			push!(L_samples,sample)
		end
		if sample > R_inf
			push!(U_samples, sample)
		end
	end

	LPM = 0.0
	UPM = 0.0
	for x in L_samples
		LPM += (R_inf - x)*(p_samples[x]/length(s))
	end
	for x in U_samples
		UPM += (x - R_inf)*(p_samples[x]/length(s))
	end
	xo = 2.0/(1+exp(-log(UPM/LPM))) - 1

	return xo
end
