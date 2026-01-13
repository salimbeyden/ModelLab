import requests
import time

BASE_URL = "http://localhost:8000/v1"

def test_pipeline():
    # 1. Upload basic CSV
    csv_content = "target,feature1,feature2\n0,1,10\n1,2,20\n0,3,10\n1,4,20"
    files = {'file': ('dataset.csv', csv_content, 'text/csv')}
    
    print("Uploading dataset...")
    res = requests.post(f"{BASE_URL}/datasets", files=files)
    if res.status_code != 200:
        print("Upload failed:", res.text)
        return
    
    ds_meta = res.json()
    ds_id = ds_meta['id']
    print(f"Dataset ID: {ds_id}")
    
    # 2. Trigger Profile (optional check)
    print("Profiling...")
    res = requests.post(f"{BASE_URL}/datasets/{ds_id}/profile")
    print("Profile:", res.json())
    
    # 3. Create Run (EBM)
    config = {
        "dataset_id": ds_id,
        "target": "target",
        "task": "classification",
        "model_id": "ebm",
        "model_params": {"max_bins": 32}
    }
    
    print("Starting Run...")
    res = requests.post(f"{BASE_URL}/runs", json=config)
    if res.status_code != 200:
        print("Run Creation Failed:", res.text)
        return
        
    run = res.json()
    run_id = run['id']
    print(f"Run ID: {run_id}")
    
    # 4. Poll Status
    max_retries = 10
    for _ in range(max_retries):
        res = requests.get(f"{BASE_URL}/runs/{run_id}")
        state = res.json()
        status = state['status']
        print(f"Status: {status}")
        
        if status in ["completed", "failed"]:
            break
        time.sleep(1)
        
    print("Final State:", state)

if __name__ == "__main__":
    test_pipeline()
