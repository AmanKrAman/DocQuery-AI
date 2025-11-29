from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import shutil
from datetime import datetime
from typing import List, Dict

router = APIRouter()

def delete_vector_collection(unique_id: str) -> bool:
    import sqlite3
    import glob
    
    try:
        vector_db_path = "data/vector_db"
        collection_name = f"documents_{unique_id}"
        db_path = os.path.join(vector_db_path, "chroma.sqlite3")
        
        if not os.path.exists(db_path):
            return False
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
            result = cursor.fetchone()
            collection_id = result[0] if result else None
        except:
            collection_id = None
        
        tables_to_clean = [
            ("embeddings_queue", "DELETE FROM embeddings_queue WHERE id IN (SELECT id FROM collections WHERE name = ?)"),
            ("segments", "DELETE FROM segments WHERE collection IN (SELECT id FROM collections WHERE name = ?)"),
            ("collections", "DELETE FROM collections WHERE name = ?")
        ]
        
        deleted_any = False
        for table_name, query in tables_to_clean:
            try:
                cursor.execute(query, (collection_name,))
                if cursor.rowcount > 0:
                    deleted_any = True
            except Exception as e:
                pass
        
        if deleted_any:
            cursor.execute("SELECT id FROM segments")
            valid_segments = set(row[0] for row in cursor.fetchall())
            
            conn.commit()
            conn.close()
            
            uuid_folders = glob.glob(os.path.join(vector_db_path, "*-*-*-*-*"))
            
            deleted_folders = 0
            for folder in uuid_folders:
                folder_name = os.path.basename(folder)
                if folder_name not in valid_segments:
                    try:
                        shutil.rmtree(folder)
                        deleted_folders += 1
                    except Exception as e:
                        pass
            
            if deleted_folders > 0:
                pass
            else:
                pass
            
            return True
        else:
            conn.close()
            return True
            
    except Exception as e:
        return False


def delete_vector_documents_by_filename(unique_id: str, filename: str) -> bool:
    import sqlite3
    
    try:
        vector_db_path = "data/vector_db"
        collection_name = f"documents_{unique_id}"
        db_path = os.path.join(vector_db_path, "chroma.sqlite3")
        
        if not os.path.exists(db_path):
            return False
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return True
        
        collection_id = result[0]
        
        cursor.execute("SELECT id FROM segments WHERE collection = ?", (collection_id,))
        segment_result = cursor.fetchone()
        
        if not segment_result:
            conn.close()
            return True
        
        segment_id = segment_result[0]
        
        cursor.execute("SELECT id, embedding_id FROM embeddings WHERE segment_id = ?", (segment_id,))
        embeddings = cursor.fetchall()
        
        embedding_ids_to_delete = []
        internal_ids_to_delete = []
        
        for internal_id, embedding_id in embeddings:
            cursor.execute("""
                SELECT id FROM embedding_metadata 
                WHERE id = ? AND key = 'filename' AND string_value = ?
            """, (internal_id, filename))
            
            if cursor.fetchone():
                embedding_ids_to_delete.append(embedding_id)
                internal_ids_to_delete.append(internal_id)
        
        if internal_ids_to_delete:
            placeholders = ','.join('?' * len(internal_ids_to_delete))
            
            cursor.execute(f"DELETE FROM embedding_metadata WHERE id IN ({placeholders})", internal_ids_to_delete)
            cursor.execute(f"DELETE FROM embeddings WHERE id IN ({placeholders})", internal_ids_to_delete)
            
            deleted_count = len(internal_ids_to_delete)
            conn.commit()
        else:
            pass
        
        conn.close()
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False

@router.get("/files/{unique_id}")
async def list_files(unique_id: str):
    try:
        folder_path = os.path.join("workfiles", unique_id)

        if not os.path.exists(folder_path):
            raise HTTPException(
                status_code=404,
                detail=f"No files found for unique_id: {unique_id}"
            )

        files = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                file_stats = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size_bytes": file_stats.st_size,
                    "size_kb": round(file_stats.st_size / 1024, 2),
                    "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                })

        return JSONResponse(content={
            "unique_id": unique_id,
            "files": files,
            "count": len(files),
            "folder_path": folder_path
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{unique_id}")
async def delete_files(unique_id: str):
    try:
        folder_path = os.path.join("workfiles", unique_id)

        if not os.path.exists(folder_path):
            raise HTTPException(
                status_code=404,
                detail=f"No files found for unique_id: {unique_id}"
            )

        file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

        shutil.rmtree(folder_path)

        vector_db_deleted = delete_vector_collection(unique_id)

        return JSONResponse(content={
            "message": f"All data for unique_id '{unique_id}' deleted successfully",
            "unique_id": unique_id,
            "files_deleted": file_count,
            "folder_deleted": True,
            "vector_db_deleted": vector_db_deleted
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{unique_id}/{filename}")
async def get_file_info(unique_id: str, filename: str):
    try:
        file_path = os.path.join("workfiles", unique_id, filename)

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File '{filename}' not found for unique_id: {unique_id}"
            )

        file_stats = os.stat(file_path)
        file_extension = os.path.splitext(filename)[1].lower()

        return JSONResponse(content={
            "unique_id": unique_id,
            "filename": filename,
            "file_path": file_path,
            "extension": file_extension,
            "size_bytes": file_stats.st_size,
            "size_kb": round(file_stats.st_size / 1024, 2),
            "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "is_readable": os.access(file_path, os.R_OK)
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{unique_id}/{filename}")
async def delete_single_file(unique_id: str, filename: str):
    try:
        file_path = os.path.join("workfiles", unique_id, filename)

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File '{filename}' not found for unique_id: {unique_id}"
            )

        file_size = os.path.getsize(file_path)

        os.remove(file_path)
        folder_path = os.path.join("workfiles", unique_id)
        folder_deleted = False
        vector_db_updated = False
        
        if not os.listdir(folder_path):
            shutil.rmtree(folder_path)
            folder_deleted = True
            vector_db_updated = delete_vector_collection(unique_id)
        else:
            vector_db_updated = delete_vector_documents_by_filename(unique_id, filename)

        return JSONResponse(content={
            "message": f"File '{filename}' deleted successfully",
            "unique_id": unique_id,
            "filename": filename,
            "size_deleted_bytes": file_size,
            "folder_deleted": folder_deleted,
            "vector_db_updated": vector_db_updated
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
