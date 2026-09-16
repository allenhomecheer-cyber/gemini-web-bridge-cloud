import os
import sys
import asyncio
import base64
import json
import uuid
import httpx
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from gemini_webapi import GeminiClient

sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Gemini Web & ComfyUI Gateway", description="Unified gateway for Gemini Imagen 3 & Veo Video Generation and Local ComfyUI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
SCRATCH_DIR = DATA_DIR
CONFIG_FILE = DATA_DIR / "gemini_web_cookies.json"
OUTPUT_DIR = DATA_DIR / "gemini_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client: Optional[GeminiClient] = None
is_initialized = False
gen_lock = asyncio.Lock()
COMFYUI_BACKEND = "http://127.0.0.1:8188"

class CookieConfig(BaseModel):
    secure_1psid: str
    secure_1psidts: str = ""
    secure_1psidcc: str = ""

class ImageGenRequest(BaseModel):
    prompt: str
    aspect_ratio: Optional[str] = "16:9"
    response_format: Optional[str] = "json"
    input_image_base64: Optional[str] = None
    input_image_url: Optional[str] = None

class VideoGenRequest(BaseModel):
    prompt: str
    aspect_ratio: Optional[str] = "16:9"
    duration_sec: Optional[int] = 5
    response_format: Optional[str] = "json" # "json" or "binary"
    input_image_base64: Optional[str] = None
    input_image_url: Optional[str] = None

async def init_gemini_client():
    global client, is_initialized
    if not CONFIG_FILE.exists():
        print("⚠️ 尚未配置 Gemini Web Cookies (gemini_web_cookies.json)")
        is_initialized = False
        return False
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        psid = cfg.get("secure_1psid", "")
        psidts = cfg.get("secure_1psidts", "")
        
        if not psid:
            print("⚠️ secure_1psid 為空")
            is_initialized = False
            return False
            
        print(f"🔄 正在初始化 Gemini Web 客戶端 (PSID: {psid[:10]}...)...")
        client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts if psidts else None)
        psidcc = cfg.get("secure_1psidcc", "")
        if psidcc:
            client._cookies.set("__Secure-1PSIDCC", psidcc, domain=".google.com", secure=True)
        await client.init()
        is_initialized = True
        print("✅ Gemini WebAPI 客戶端初始化成功！")
        return True
    except Exception as e:
        print(f"❌ Gemini WebAPI 初始化失敗: {e}")
        is_initialized = False
        return False

PROFILE_DIR = SCRATCH_DIR / "browser_profile"

async def auto_refresh_cookies_headless():
    """在背景無頭模式下自動啟動 Playwright 讀取最新 Cookie 並更新客戶端"""
    try:
        from playwright.async_api import async_playwright
        print("🔄 正在嘗試於背景無頭模式自動刷新 Google Gemini Cookies...")
        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=str(PROFILE_DIR),
                headless=True,
                channel="chrome",
                args=["--disable-blink-features=AutomationControlled"]
            )
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto("https://gemini.google.com/app", timeout=15000)
            await asyncio.sleep(2)
            cookies = await context.cookies(["https://gemini.google.com", "https://google.com"])
            cookie_dict = {c["name"]: c["value"] for c in cookies}
            await context.close()
            
            psid = cookie_dict.get("__Secure-1PSID")
            psidts = cookie_dict.get("__Secure-1PSIDTS")
            psidcc = cookie_dict.get("__Secure-1PSIDCC")
            
            if psid:
                print(f"✨ 背景成功擷取到最新 Gemini Cookie: PSID={psid[:10]}...")
                cfg = {"secure_1psid": psid, "secure_1psidts": psidts or "", "secure_1psidcc": psidcc or ""}
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                return await init_gemini_client()
            else:
                print("⚠️ 背景無頭瀏覽器尚未登入 Google 帳號，請執行桌面同步工具登入一次")
    except Exception as e:
        print(f"⚠️ 背景自動刷新 Cookie 失敗: {e}")
    return False

async def background_cookie_refresher():
    """每 30 分鐘自動於背景執行一次 Cookie 刷新維持 Session 活躍"""
    while True:
        await asyncio.sleep(1800)
        try:
            await auto_refresh_cookies_headless()
        except Exception:
            pass

@app.on_event("startup")
async def startup_event():
    await init_gemini_client()
    asyncio.create_task(background_cookie_refresher())

@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "gemini_authenticated": is_initialized,
        "config_exists": CONFIG_FILE.exists(),
        "model": "Imagen 3 & Veo via Gemini Web Session",
        "comfyui_backend": COMFYUI_BACKEND
    }

@app.post("/cookies")
async def set_cookies(cookie_data: CookieConfig):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cookie_data.dict(), f, indent=2)
    
    success = await init_gemini_client()
    if not success:
        raise HTTPException(status_code=400, detail="Cookies 驗證失敗，請檢查 __Secure-1PSID 與 __Secure-1PSIDTS 是否有效")
    return {"status": "success", "message": "Gemini Web Cookies 已成功更新並驗證通過！"}

@app.post("/v1/images/generate")
async def generate_image(req: ImageGenRequest):
    global client, is_initialized
    if not is_initialized or not client:
        if not await init_gemini_client():
            raise HTTPException(status_code=401, detail="Gemini Web 尚未完成 Cookie 驗證，請先於 /cookies 填入 Cookie")
    
    temp_files_to_cleanup = []
    async with gen_lock:
        try:
            files = []
            if req.input_image_base64:
                temp_img_path = SCRATCH_DIR / f"input_ref_{uuid.uuid4().hex[:8]}.jpg"
                b64_data = req.input_image_base64
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                with open(temp_img_path, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                files.append(temp_img_path)
                temp_files_to_cleanup.append(temp_img_path)
            elif req.input_image_url:
                temp_img_path = SCRATCH_DIR / f"input_ref_{uuid.uuid4().hex[:8]}.jpg"
                async with httpx.AsyncClient(timeout=30) as http:
                    r = await http.get(req.input_image_url)
                    if r.status_code == 200:
                        with open(temp_img_path, "wb") as f:
                            f.write(r.content)
                        files.append(temp_img_path)
                        temp_files_to_cleanup.append(temp_img_path)

            prompt_prefix = "Generate an image"
            if files:
                prompt_prefix = "Based on the provided reference image, generate an image"
                
            full_prompt = f"{prompt_prefix}: {req.prompt}"
            if req.aspect_ratio:
                full_prompt += f", aspect ratio {req.aspect_ratio}"
                
            print(f"🎨 發送生圖指令至 Gemini Web (附加參考圖: {len(files)}): {full_prompt[:90]}...")
            
            response = await client.generate_content(
                prompt=full_prompt,
                files=files if files else None
            )
            
            if not response.images or len(response.images) == 0:
                return JSONResponse(status_code=422, content={
                    "error": "Gemini 未回傳生成圖片",
                    "text_response": response.text
                })
                
            img = response.images[0]
            gen_id = uuid.uuid4().hex[:10]
            filename = f"gemini_gen_{gen_id}.jpg"
            
            saved_path = await img.save(
                path=str(OUTPUT_DIR),
                filename=filename,
                verbose=True
            )
            
            with open(saved_path, "rb") as f:
                img_bytes = f.read()
                
            if req.response_format == "binary":
                return Response(content=img_bytes, media_type="image/jpeg", headers={
                    "X-Gemini-Title": getattr(img, 'title', 'gemini_generated.jpg'),
                    "X-Gemini-Filename": filename
                })
                
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            
            return {
                "status": "success",
                "model": "Imagen 3 via Gemini Web",
                "file_name": filename,
                "file_path": str(saved_path),
                "file_size": len(img_bytes),
                "url": getattr(img, 'url', ''),
                "text": response.text,
                "b64_json": img_b64,
                "data": [
                    {
                        "b64_json": img_b64,
                        "url": getattr(img, 'url', '')
                    }
                ]
            }
        except Exception as e:
            print(f"❌ 生圖失敗: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            for p in temp_files_to_cleanup:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

@app.post("/v1/videos/generate")
async def generate_video(req: VideoGenRequest):
    global client, is_initialized
    if not is_initialized or not client:
        if not await init_gemini_client():
            raise HTTPException(status_code=401, detail="Gemini Web 尚未完成 Cookie 驗證，請先於 /cookies 填入 Cookie")
    
    temp_files_to_cleanup = []
    async with gen_lock:
        try:
            files = []
            if req.input_image_base64:
                temp_img_path = SCRATCH_DIR / f"input_ref_{uuid.uuid4().hex[:8]}.jpg"
                b64_data = req.input_image_base64
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                with open(temp_img_path, "wb") as f:
                    f.write(base64.b64decode(b64_data))
                files.append(temp_img_path)
                temp_files_to_cleanup.append(temp_img_path)
            elif req.input_image_url:
                temp_img_path = SCRATCH_DIR / f"input_ref_{uuid.uuid4().hex[:8]}.jpg"
                async with httpx.AsyncClient(timeout=30) as http:
                    r = await http.get(req.input_image_url)
                    if r.status_code == 200:
                        with open(temp_img_path, "wb") as f:
                            f.write(r.content)
                        files.append(temp_img_path)
                        temp_files_to_cleanup.append(temp_img_path)

            prompt_prefix = "Generate a video"
            if files:
                prompt_prefix = "Generate a video based on this image"
                
            full_prompt = f"{prompt_prefix}: {req.prompt}"
            if req.aspect_ratio:
                full_prompt += f", aspect ratio {req.aspect_ratio}"
            if req.duration_sec:
                full_prompt += f", duration {req.duration_sec}s"
                
            print(f"🎬 發送生影指令至 Gemini Web Veo (附加參考圖: {len(files)}): {full_prompt[:90]}...")
            
            response = await client.generate_content(
                prompt=full_prompt,
                files=files if files else None
            )
            
            if not response.videos or len(response.videos) == 0:
                return JSONResponse(status_code=422, content={
                    "error": "Gemini 未回傳生成影片",
                    "text_response": response.text
                })
                
            vid = response.videos[0]
            gen_id = uuid.uuid4().hex[:10]
            filename = f"gemini_veo_{gen_id}.mp4"
            
            save_res = await vid.save(
                path=str(OUTPUT_DIR),
                filename=filename,
                verbose=True
            )
            
            video_path = save_res.get("video") if isinstance(save_res, dict) else str(save_res)
            
            if not video_path or not os.path.exists(video_path):
                raise HTTPException(status_code=500, detail="影片下載失敗")
                
            with open(video_path, "rb") as f:
                vid_bytes = f.read()
                
            if req.response_format == "binary":
                return Response(content=vid_bytes, media_type="video/mp4", headers={
                    "X-Gemini-Title": getattr(vid, 'title', 'gemini_veo.mp4'),
                    "X-Gemini-Filename": filename
                })
                
            vid_b64 = base64.b64encode(vid_bytes).decode("utf-8")
            
            return {
                "status": "success",
                "model": "Google Veo via Gemini Web",
                "file_name": filename,
                "file_path": str(video_path),
                "file_size": len(vid_bytes),
                "url": getattr(vid, 'url', ''),
                "text": response.text,
                "b64_json": vid_b64
            }
        except Exception as e:
            err_str = str(e)
            if "UNAUTHENTICATED" in err_str or "expired" in err_str or "Permission denied" in err_str:
                print("⚠️ 檢測到 Gemini Session 過期，正在嘗試背景無感自動刷新 Cookie 並重新生成...")
                refreshed = await auto_refresh_cookies_headless()
                if refreshed and client:
                    try:
                        print("🔄 刷新成功，正在重新發送生影請求...")
                        response = await client.generate_content(
                            prompt=full_prompt,
                            files=files if files else None
                        )
                        if response.videos and len(response.videos) > 0:
                            vid = response.videos[0]
                            gen_id = uuid.uuid4().hex[:10]
                            filename = f"gemini_veo_{gen_id}.mp4"
                            save_res = await vid.save(
                                path=str(OUTPUT_DIR),
                                filename=filename,
                                verbose=True
                            )
                            video_path = save_res.get("video") if isinstance(save_res, dict) else str(save_res)
                            with open(video_path, "rb") as f:
                                vid_bytes = f.read()
                            vid_b64 = base64.b64encode(vid_bytes).decode("utf-8")
                            return {
                                "status": "success",
                                "model": "Google Veo via Gemini Web (Auto-Refreshed)",
                                "file_name": filename,
                                "file_path": str(video_path),
                                "file_size": len(vid_bytes),
                                "url": getattr(vid, 'url', ''),
                                "text": response.text,
                                "b64_json": vid_b64
                            }
                    except Exception as retry_err:
                        print(f"❌ 重試生影失敗: {retry_err}")
            print(f"❌ 生影失敗: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            for p in temp_files_to_cleanup:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

# 透傳 ComfyUI 請求 (如 /prompt, /history, /view, /upload/image 等)
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_comfyui(path_name: str, request: Request):
    if path_name.startswith("v1/") or path_name in ["health", "cookies"]:
        raise HTTPException(status_code=404, detail="Not Found")
    
    url = f"{COMFYUI_BACKEND}/{path_name}"
    if request.query_params:
        url += f"?{request.query_params}"
        
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    
    async with httpx.AsyncClient(timeout=120.0) as http:
        try:
            r = await http.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body
            )
            return Response(
                content=r.content,
                status_code=r.status_code,
                headers=dict(r.headers)
            )
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": f"Failed to proxy to ComfyUI: {e}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
