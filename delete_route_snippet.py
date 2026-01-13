@app.delete("/datasets/{id}", status_code=204)
def delete_dataset(id: str):
    try:
        data_service.delete_dataset(id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
