function primal_dual(c,A,b,x0,p0; epsilon=1e-9, max_iter=1000,alpha=0.9)

    # Preparing variables for the trajectories
    x1_traj = []
    x2_traj = []

    # Initialization
    x = x0
    p = p0
    s = c-A'*p
    n = size(A,2)
    m = size(b,1)
    e = ones(size(x,1),1)
    rho = 0.5
    for i in 1:max_iter
      # Recording the trajectories of x1 and x2
      push!(x1_traj, x[1])
      push!(x2_traj, x[2])
        
      # Newton's Step
      mu = rho*dot(x,s)/n
      X = Diagonal(x)
      S = Diagonal(s)
      LHS =  [A zeros(m,m) zeros(m,n);
                 zeros(n,n) A' I;
                S zeros(n,m) X]
      RHS = [zeros(n,1); zeros(m,1); mu*e-X*S*e]
      d = LHS\RHS
      dx = d[1:n]
#       println(d)
      dp = d[n+1:n+m]
      ds = d[n+m+1:end]
        
      # Calculate Steps Size
      beta_p = minimum([1 ;-alpha*x[dx.<0]./dx[dx.<0]])
      beta_d = minimum([1 ;-alpha*s[ds.<0]./ds[ds.<0]])
        
      # Optimality check
      if dot(x,s) < epsilon
        break
      end

      # Update
      x = x+beta_p*dx
      p = p+beta_d*dp
      s = s+beta_d*ds
    end

    return x1_traj, x2_traj
end