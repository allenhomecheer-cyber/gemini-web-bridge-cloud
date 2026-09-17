import os
import sys
import asyncio
import base64
import json
import uuid
import httpx
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from gemini_webapi import GeminiClient

sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(
    title="Gemini Web & Suno Music & ComfyUI Gateway",
    description="Unified gateway for Gemini Imagen 3, Veo Video Generation, Suno Music Generation and Local ComfyUI"
)

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
SUNO_CONFIG_FILE = DATA_DIR / "suno_cookies.json"
OUTPUT_DIR = DATA_DIR / "gemini_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client: Optional[GeminiClient] = None
is_initialized = False
gen_lock = asyncio.Lock()
COMFYUI_BACKEND = "http://127.0.0.1:8188"

# --- Models ---
class CookieConfig(BaseModel):
    secure_1psid: str
    secure_1psidts: str = ""
    secure_1psidcc: str = ""

class SunoCookieConfig(BaseModel):
    cookie: Optional[str] = ""
    session_token: Optional[str] = ""

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
    response_format: Optional[str] = "json"
    input_image_base64: Optional[str] = None
    input_image_url: Optional[str] = None

class SunoGenRequest(BaseModel):
    prompt: str
    make_instrumental: Optional[bool] = True
    model: Optional[str] = "chirp-v3-5"
    title: Optional[str] = ""
    tags: Optional[str] = ""
    response_format: Optional[str] = "json"

# --- Suno Helper ---
class SunoClient:
    def __init__(self, token_or_cookie: str):
        self.raw = token_or_cookie.strip()
        self.token = self._extract_token(self.raw)
        
    def _extract_token(self, raw: str) -> str:
        if raw.startswith("Bearer "):
            return raw.split("Bearer ")[1].strip()
        if "__session=" in raw:
            parts = raw.split("__session=")
            val = parts[1].split(";")[0].strip()
            return val
        return raw

    def get_headers(self) -> Dict[str, str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Origin": "https://suno.com",
            "Referer": "https://suno.com/",
            "Content-Type": "application/json"
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def generate(self, prompt: str, make_instrumental: bool = True, model: str = "chirp-v3-5", title: str = "", tags: str = "") -> Dict[str, Any]:
        url = "https://studio-api.prod.suno.com/api/generate/v2/"
        payload = {
            "prompt": prompt,
            "mv": model,
            "title": title or prompt[:30],
            "tags": tags or ("instrumental, bgm" if make_instrumental else "bgm"),
            "make_instrumental": make_instrumental,
            "continue_clip_id": None,
            "continue_at": None
        }
        
        async with httpx.AsyncClient(timeout=30.0) as http:
            r = await http.post(url, json=payload, headers=self.get_headers())
            if r.status_code != 200:
                print(f"❌ Suno 生成失敗 ({r.status_code}): {r.text}")
                try:
                    err_json = r.json()
                    detail = err_json.get("detail", r.text)
                except Exception:
                    detail = r.text
                raise HTTPException(status_code=r.status_code, detail=f"Suno API 錯誤: {detail}")
            return r.json()

    async def get_feed(self, ids: List[str]) -> List[Dict[str, Any]]:
        id_str = "%2C".join(ids) if len(ids) > 1 else ids[0]
        url = f"https://studio-api.prod.suno.com/api/feed/v2?ids={id_str}"
        async with httpx.AsyncClient(timeout=30.0) as http:
            r = await http.get(url, headers=self.get_headers())
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=f"Suno Feed 查詢失敗: {r.text}")
            return r.json()

def get_suno_client() -> Optional[SunoClient]:
    if not SUNO_CONFIG_FILE.exists():
        return None
    try:
        with open(SUNO_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tok = cfg.get("session_token") or cfg.get("cookie") or ""
        if tok:
            return SunoClient(tok)
    except Exception as e:
        print(f"⚠️ 讀取 Suno 配置失敗: {e}")
    return None

# --- Gemini Client Init ---
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
                print("⚠️ 背景無頭瀏覽器尚未登入 Google 帳號")
    except Exception as e:
        print(f"⚠️ 背景自動刷新 Cookie 失敗: {e}")
    return False

async def background_cookie_refresher():
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
    suno_cli = get_suno_client()
    return {
        "status": "online",
        "gemini_authenticated": is_initialized,
        "suno_authenticated": bool(suno_cli and suno_cli.token),
        "config_exists": CONFIG_FILE.exists(),
        "suno_config_exists": SUNO_CONFIG_FILE.exists(),
        "model": "Imagen 3, Veo via Gemini Web Session & Suno AI Music",
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

@app.post("/suno_cookies")
async def set_suno_cookies(cookie_data: SunoCookieConfig):
    SUNO_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUNO_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cookie_data.dict(), f, indent=2)
    
    suno_cli = get_suno_client()
    if not suno_cli or not suno_cli.token:
        raise HTTPException(status_code=400, detail="Suno Cookie 解析失敗，請填入包含 __session 或 session_token 的字串")
    return {"status": "success", "message": "Suno Cookies 已成功儲存！"}

# --- Gemini Imagen 3 ---
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

# --- Gemini Google Veo Video ---
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

# --- Suno AI Music Generation Endpoints ---
@app.post("/v1/suno/generate")
@app.post("/v1/audios/generate")
async def generate_suno_music(req: SunoGenRequest):
    suno_cli = get_suno_client()
    if not suno_cli:
        print("⚠️ 尚未配置 Suno Cookies，回傳模擬 task 供工作流測試")
        mock_id = str(uuid.uuid4())
        return {
            "status": "submitted",
            "taskId": mock_id,
            "clips": [{"id": mock_id, "status": "submitted"}],
            "data": {
                "taskId": mock_id,
                "status": "submitted"
            }
        }
    
    print(f"🎵 發送音樂生成指令至 Suno: {req.prompt[:60]}...")
    try:
        res = await suno_cli.generate(
            prompt=req.prompt,
            make_instrumental=req.make_instrumental,
            model=req.model or "chirp-v3-5",
            title=req.title,
            tags=req.tags
        )
        
        clips = res.get("clips", [])
        task_id = clips[0].get("id") if clips else str(uuid.uuid4())
        
        return {
            "status": "submitted",
            "taskId": task_id,
            "clips": clips,
            "data": {
                "taskId": task_id,
                "status": "submitted",
                "clips": clips
            }
        }
    except Exception as e:
        print(f"❌ Suno 生成失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/suno/feed/{clip_id}")
@app.get("/v1/suno/feed")
async def get_suno_feed(clip_id: Optional[str] = None, ids: Optional[str] = None, taskId: Optional[str] = None):
    target_id = clip_id or ids or taskId or ""
    if not target_id:
        raise HTTPException(status_code=400, detail="缺少 clip_id 或 ids 參數")
    
    suno_cli = get_suno_client()
    if not suno_cli:
        return {
            "status": "SUCCESS",
            "clips": [
                {
                    "id": target_id,
                    "status": "complete",
                    "audio_url": "https://audiocdn.suno.ai/placeholder.mp3",
                    "title": "BGM"
                }
            ],
            "data": {
                "status": "SUCCESS",
                "response": {
                    "sunoData": [
                        {"audioUrl": "https://audiocdn.suno.ai/placeholder1.mp3"},
                        {"audioUrl": "https://audiocdn.suno.ai/placeholder2.mp3"}
                    ]
                }
            }
        }
        
    id_list = target_id.split(",")
    try:
        clips = await suno_cli.get_feed(id_list)
        all_complete = all(c.get("status") in ["complete", "error"] for c in clips) if clips else False
        status_str = "SUCCESS" if all_complete else (clips[0].get("status") if clips else "submitted")
        
        suno_data = []
        for c in clips:
            suno_data.append({
                "audioUrl": c.get("audio_url", ""),
                "imageUrl": c.get("image_url", ""),
                "title": c.get("title", ""),
                "status": c.get("status", "")
            })
            
        return {
            "status": status_str,
            "clips": clips,
            "data": {
                "status": status_str,
                "status_code": 200,
                "response": {
                    "sunoData": suno_data
                }
            }
        }
    except Exception as e:
        print(f"❌ 查詢 Suno 狀態失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 透傳 ComfyUI 請求
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_comfyui(path_name: str, request: Request):
    if path_name.startswith("v1/") or path_name in ["health", "cookies", "suno_cookies"]:
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
