import torch
import torch.nn as pt


def train_model(model, train_loader, validation_loader, epochs=5,
                learning_rate=0.001):
    # Entrena por lotes y devuelve el modelo con sus pesos actualizados.
    calculate_loss = pt.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_images = 0

        for images, labels in train_loader:
            optimizer.zero_grad()  # Borra los gradientes del lote anterior.
            outputs = model(images)
            loss = calculate_loss(outputs, labels)
            loss.backward()  # Calcula cómo influye cada peso en el error.
            optimizer.step()  # Actualiza los pesos para reducir el error.

            total_loss += loss.item() * len(images)
            total_images += len(images)

        model.eval()
        correct = 0
        total_validation = 0
        # Validación: sin gradientes ni actualizaciones de pesos.
        with torch.no_grad():
            for images, labels in validation_loader:
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total_validation += len(images)

        mean_loss = total_loss / total_images
        accuracy = 100 * correct / total_validation
        print(f"Época {epoch + 1}/{epochs} - "
              f"Pérdida: {mean_loss:.4f} - "
              f"Accuracy Validation: {accuracy:.2f}%")

    return model
