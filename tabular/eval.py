def evaluate(model, val_loader):
    model.eval()
    all_preds = []
    all_truths = []
    for i, (x, y) in tqdm(enumerate(val_loader), desc="Evaluating"):
        with torch.no_grad():
            out = torch.argmax(F.softmax(model(x), dim=1), dim=1).cpu().numpy()
            truths = y.cpu().numpy()
            all_preds.extend(out)
            all_truths.extend(truths)

    accuracy = accuracy_score(all_truths, all_preds)
    return accuracy