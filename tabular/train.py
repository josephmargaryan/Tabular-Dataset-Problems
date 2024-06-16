def train(model, num_epochs, train_loader, val_loader, lr=1e-4, weight_decay=1e-5, patience=25):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.02)
    train_losses = []
    val_losses = []
    best_model = None
    counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        
        model.eval()
        val_loss = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                out = model(x)
                loss = criterion(out, y)
                val_loss.append(loss.item())
        val_loss = np.mean(val_loss)
        
        if best_val_loss > val_loss:
            counter = 0
            best_val_loss = val_loss
            best_model = model.state_dict()
        else:
            counter += 1
            if counter > patience:
                print("Early stopping")
                break
        scheduler.step()
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return best_model, best_val_loss