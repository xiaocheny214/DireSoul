/**
 * Quick-create 专用的后端直连客户端。
 *
 * 有意绕开 `@/app/adapters/generation.ts` 的有损类型联合（后端
 * `character_image`/`character_action` 任务类型 → 前端
 * `character_template`/`first_frame`/`complete_animation` 的强制转换，
 * 以及 `image_urls`/`frames` → `images`/`frames[{url}]` 的字段改名会丢
 * `root_motion` 等信息）。这里只做最薄的一层：直接说后端原生的字段名，
 * 不做任何联合类型翻译。
 */

import { get, post } from '@/app/adapters/http-client'

/* ─── 请求参数（调用方视角，驼峰） ─── */

export interface SubmitImageParams {
  projectId: number
  prompt: string
  referenceImageUrl?: string | null
  width: number
  height: number
}

export interface SubmitActionParams {
  projectId: number
  characterId: number
  actionType: string
  referenceImageUrls: string[]
  numFrames?: number
}

export interface PollTaskOptions {
  intervalMs?: number
  timeoutMs?: number
}

/* ─── 后端原生响应（GenerationTaskOut，字段名与后端一致） ─── */

export interface BackendTaskResult {
  status: string
  result: Record<string, unknown> | null
  error_message: string | null
}

interface BackendGenerationTaskOut extends BackendTaskResult {
  id: number
  user_id: number
  project_id: number | null
  task_type: string
  input_payload: Record<string, unknown> | null
}

const DEFAULT_POLL_INTERVAL_MS = 2000
const DEFAULT_POLL_TIMEOUT_MS = 300_000

/**
 * 提交角色图片生成任务：POST /generation/image。
 * 返回后端任务 id（仅此一个字段，调用方自行轮询）。
 */
export async function submitImage(params: SubmitImageParams): Promise<{ id: number }> {
  const raw = await post<BackendGenerationTaskOut>('/generation/image', {
    user_id: 1,
    project_id: params.projectId,
    prompt: params.prompt,
    reference_image_url: params.referenceImageUrl ?? null,
    width: params.width,
    height: params.height,
    num_images: 1,
  })
  return { id: raw.id }
}

/**
 * 提交角色动作生成任务：POST /generation/action。
 */
export async function submitAction(params: SubmitActionParams): Promise<{ id: number }> {
  const raw = await post<BackendGenerationTaskOut>('/generation/action', {
    user_id: 1,
    project_id: params.projectId,
    character_id: params.characterId,
    action_type: params.actionType,
    reference_image_urls: params.referenceImageUrls,
    num_frames: params.numFrames ?? 16,
  })
  return { id: raw.id }
}

/**
 * 轮询 GET /generation/tasks/{id}，直到 status 落到终态
 * （completed/failed）或超时。
 *
 * 注意：后端该接口要求 `project_id` query 参数（校验 gt=0），
 * 因此 `projectId` 是必传参数，而非可选项。
 */
export async function pollTask(
  taskId: number,
  projectId: number,
  options?: PollTaskOptions,
): Promise<BackendTaskResult> {
  const intervalMs = options?.intervalMs ?? DEFAULT_POLL_INTERVAL_MS
  const timeoutMs = options?.timeoutMs ?? DEFAULT_POLL_TIMEOUT_MS
  const deadline = Date.now() + timeoutMs

  for (;;) {
    const raw = await get<BackendGenerationTaskOut>(
      `/generation/tasks/${taskId}?project_id=${encodeURIComponent(String(projectId))}`,
    )
    if (raw.status === 'completed' || raw.status === 'failed') {
      return { status: raw.status, result: raw.result, error_message: raw.error_message }
    }
    if (Date.now() >= deadline) {
      return {
        status: 'failed',
        result: null,
        error_message: `轮询超时(${timeoutMs}ms)：任务 ${taskId} 仍处于 ${raw.status}`,
      }
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }
}
