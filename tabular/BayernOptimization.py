from bayes_opt import BayesianOptimization
import time

def evaluate_network(hidden_dim):
    hidden_dim = int(hidden_dim) 

    
    model = BaseLine(in_channels=X.shape[1], hidden_dim=hidden_dim, num_classes=3)
    
    try:
        _, val_loss = train(model, num_epochs=20, train_loader=train_loader, val_loader=val_loader)
    except Exception as e:
        print(f"Exception: {e}")
        return float('inf')
    
    return -val_loss

# Define the parameter bounds
pbounds = {
    'hidden_dim': (50, 150)
}

# Perform Bayesian Optimization
optimizer = BayesianOptimization(
    f=evaluate_network,
    pbounds=pbounds,
    verbose=2,  
    random_state=1,
)

start_time = time.time()
optimizer.maximize(init_points=2, n_iter=6) # 2 random and 6 bayesian iterations
time_took = time.time() - start_time

def hms_string(sec_elapsed):
    h = int(sec_elapsed // 3600)
    m = int((sec_elapsed % 3600) // 60)
    s = int(sec_elapsed % 60)
    return f"{h:02}:{m:02}:{s:02}"

print(f"Total runtime: {hms_string(time_took)}")
print("Best trial:")
print(optimizer.max)