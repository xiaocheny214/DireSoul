/**
 * Quick-create 编排器（C6/C7）。
 *
 * 用一条精简的异步管线，把「一句话 prompt」跑成一个可试玩的角色：
 * 建项目 → 生成母版图 → 建角色 → 并行生成 idle/walk 动作 → 回写 → 完成。
 * 它绕开了 8 步 workflow-controller，只经过两个后端生成端点
 * (`/generation/image`、`/generation/action`) 和 `/characters`，
 * 全程说后端原生形状（通过 C5 的 backend-client），不碰有损的 GenerationType 联合。
 *
 * 对外只暴露一个可观察句柄：`start(prompt)` 立刻返回 Handle，管线在后台推进，
 * 每次相位变化都推给订阅者；视图 `subscribe` 后据 `getState()` 渲染进度，
 * 落到 `done` 时自行导航到 `/playtest/{characterId}/{outfitId}`。
 */

import type {
  ActionType,
  Character,
  CharacterApis,
  CreateCharacterOutfitInput,
  Frame,
  ProjectApis,
} from '@/entities'

import type {
  BackendTaskResult,
  PollTaskOptions,
  SubmitActionParams,
  SubmitImageParams,
} from './backend-client'

/* ─── 依赖注入形状 ─── */

/** 编排器只依赖 C5 backend-client 的这三个函数；测试可注入替身。 */
export interface QuickCreateBackendClient {
  submitImage(params: SubmitImageParams): Promise<{ id: number }>
  submitAction(params: SubmitActionParams): Promise<{ id: number }>
  pollTask(
    taskId: number,
    projectId: number,
    options?: PollTaskOptions,
  ): Promise<BackendTaskResult>
}

export interface QuickCreateOrchestratorDeps {
  projectApis: ProjectApis
  characterApis: CharacterApis
  backendClient: QuickCreateBackendClient
}

/* ─── 可观察状态 ─── */

/** 管线相位；线性推进，done/failed 为终态。 */
export type QuickCreatePhase =
  | 'preparing-project'
  | 'generating-image'
  | 'creating-character'
  | 'generating-actions'
  | 'saving'
  | 'done'
  | 'failed'

export type ActionRunStatus = 'pending' | 'running' | 'completed' | 'failed'

/** generating-actions 相位下，每个 MVP 动作的独立进度。 */
export interface ActionProgress {
  actionType: ActionType
  name: string
  status: ActionRunStatus
}

export interface QuickCreateState {
  phase: QuickCreatePhase
  /** 已建项目 ID（后端整型），未建时为 null。 */
  projectId: number | null
  /** 已建角色 ID（后端整型），done 时非空。 */
  characterId: number | null
  /** 客户端自铸的造型 UUID，done 时非空；playtest 路由用它。 */
  outfitId: string | null
  /** 生成好的母版图 URL。 */
  masterImageUrl: string | null
  /** 每个 MVP 动作的进度快照。 */
  actions: ActionProgress[]
  /** failed 相位下的人类可读原因；其余相位为 null。 */
  message: string | null
}

/** start() 返回的句柄：读当前状态 + 订阅后续变化。 */
export interface QuickCreateHandle {
  getState(): QuickCreateState
  /**
   * 注册相位变化监听；返回退订函数。
   * 订阅不会立刻回放当前值——调用方先用 getState() 取初值，再 subscribe 追后续。
   */
  subscribe(listener: (state: QuickCreateState) => void): () => void
}

export interface QuickCreateOrchestrator {
  start(prompt: string): QuickCreateHandle
}

/* ─── 配置常量 ─── */

/** MVP 只生成待机 + 行走两个动作，其余动作后续追加即可。 */
const MVP_ACTIONS: { actionType: ActionType; name: string; loop: boolean }[] = [
  { actionType: 'idle', name: '待机', loop: true },
  { actionType: 'walk', name: '行走', loop: true },
]

/** 母版尺寸；必须与建项目时的 spriteSize 一致，否则后端 _validate_project_size 会 400（G1）。 */
const SPRITE_SIZE = 256

/** 每个动作生成的帧数；测试期压到 8 以省视频成本/时长。 */
const NUM_FRAMES = 8

/** 单个任务的轮询超时（毫秒）；卡住的视频任务会以 failed 收口，而非无限轮询。 */
const PER_TASK_TIMEOUT_MS = 300_000

/** 缺省帧率回退；当帧未带 duration_ms 时使用。 */
const DEFAULT_FPS = 12

/* ─── 后端原生结果的安全提取 ─── */

interface RawBackendFrame {
  index?: number
  image_url?: unknown
  duration_ms?: unknown
  root_motion?: { dx?: unknown; dy?: unknown } | null
}

/** 从 image 任务结果里取第一张母版图；取不到返回 null。 */
function extractMasterUrl(result: Record<string, unknown> | null): string | null {
  if (!result) return null
  const urls = result.image_urls
  if (Array.isArray(urls) && urls.length > 0 && typeof urls[0] === 'string') {
    return urls[0]
  }
  return null
}

/** 从 action 任务结果里取有序帧，映射为前端 Frame（驼峰、含 rootMotion）。 */
function extractFrames(result: Record<string, unknown> | null): Frame[] {
  if (!result) return []
  const rawFrames = result.frames
  if (!Array.isArray(rawFrames)) return []
  return (rawFrames as RawBackendFrame[])
    .slice()
    .sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
    .filter((f): f is RawBackendFrame & { image_url: string } => typeof f.image_url === 'string')
    .map((f) => {
      const rm = f.root_motion
      const rootMotion =
        rm && typeof rm.dx === 'number' && typeof rm.dy === 'number'
          ? { dx: rm.dx, dy: rm.dy }
          : null
      return {
        imageUrl: f.image_url,
        durationMs: typeof f.duration_ms === 'number' ? f.duration_ms : null,
        rootMotion,
      }
    })
}

/** 由首帧时长推算帧率；无时长信息则回退默认。 */
function fpsFromFrames(frames: Frame[]): number {
  const first = frames[0]
  if (first && typeof first.durationMs === 'number' && first.durationMs > 0) {
    return Math.max(1, Math.round(1000 / first.durationMs))
  }
  return DEFAULT_FPS
}

/* ─── 工厂 ─── */

export function createQuickCreateOrchestrator(
  deps: QuickCreateOrchestratorDeps,
): QuickCreateOrchestrator {
  const { projectApis, characterApis, backendClient } = deps

  return {
    start(prompt: string): QuickCreateHandle {
      const listeners = new Set<(state: QuickCreateState) => void>()

      let state: QuickCreateState = {
        phase: 'preparing-project',
        projectId: null,
        characterId: null,
        outfitId: null,
        masterImageUrl: null,
        actions: MVP_ACTIONS.map((a) => ({
          actionType: a.actionType,
          name: a.name,
          status: 'pending' as ActionRunStatus,
        })),
        message: null,
      }

      function emit(): void {
        const snapshot = state
        for (const listener of listeners) listener(snapshot)
      }

      function setState(patch: Partial<QuickCreateState>): void {
        state = { ...state, ...patch }
        emit()
      }

      function setActionStatus(actionType: ActionType, status: ActionRunStatus): void {
        setState({
          actions: state.actions.map((a) =>
            a.actionType === actionType ? { ...a, status } : a,
          ),
        })
      }

      function fail(message: string): void {
        setState({ phase: 'failed', message })
      }

      async function run(): Promise<void> {
        // ── 步骤 1：建项目（256×256）
        setState({ phase: 'preparing-project' })
        const project = await projectApis.create({
          name: prompt.slice(0, 24).trim() || '快速创建',
          perspective: 'side',
          directionalMovement: 'single',
          spriteSize: { width: SPRITE_SIZE, height: SPRITE_SIZE },
          gameStyle: prompt,
        })
        const projectId = Number(project.id)
        setState({ projectId })

        // ── 步骤 2：生成母版图（显式带 256×256，规避 G1）
        setState({ phase: 'generating-image' })
        const imageTask = await backendClient.submitImage({
          projectId,
          prompt,
          width: SPRITE_SIZE,
          height: SPRITE_SIZE,
        })
        const imageResult = await backendClient.pollTask(imageTask.id, projectId, {
          timeoutMs: PER_TASK_TIMEOUT_MS,
        })
        if (imageResult.status !== 'completed') {
          fail(imageResult.error_message ?? '母版图生成失败')
          return
        }
        const masterUrl = extractMasterUrl(imageResult.result)
        if (!masterUrl) {
          fail('母版图任务已完成但未返回图片 URL')
          return
        }
        setState({ masterImageUrl: masterUrl })

        // ── 步骤 3：建角色（带最小 character_data：一套造型、空动作、客户端铸 outfitId）
        setState({ phase: 'creating-character' })
        const outfitId = crypto.randomUUID()
        const initialOutfit: CreateCharacterOutfitInput = {
          id: outfitId,
          name: '默认造型',
          previewUrl: masterUrl,
          actions: [],
        }
        const created: Character = await characterApis.create({
          projectId: String(projectId),
          description: prompt,
          referenceImageUrl: masterUrl,
          characterData: { outfits: [initialOutfit] },
        })
        const characterId = Number(created.id)
        // 后端会原样保留我们铸的 outfitId；兜底用返回值。
        const resolvedOutfitId = created.outfits[0]?.id ?? outfitId
        setState({ characterId })

        // ── 步骤 4：并行生成 idle / walk
        setState({ phase: 'generating-actions' })
        const actionResults = await Promise.all(
          MVP_ACTIONS.map(async (cfg) => {
            setActionStatus(cfg.actionType, 'running')
            const task = await backendClient.submitAction({
              projectId,
              characterId,
              actionType: cfg.actionType,
              referenceImageUrls: [masterUrl],
              numFrames: NUM_FRAMES,
            })
            const result = await backendClient.pollTask(task.id, projectId, {
              timeoutMs: PER_TASK_TIMEOUT_MS,
            })
            const frames = result.status === 'completed' ? extractFrames(result.result) : []
            const ok = result.status === 'completed' && frames.length > 0
            setActionStatus(cfg.actionType, ok ? 'completed' : 'failed')
            return { cfg, result, frames, ok }
          }),
        )

        const firstFailed = actionResults.find((r) => !r.ok)
        if (firstFailed) {
          fail(
            firstFailed.result.error_message ??
              `动作「${firstFailed.cfg.name}」生成失败或未返回帧`,
          )
          return
        }

        // ── 步骤 5：组装 character_data.outfits[0].actions[]（含 root_motion）→ 回写
        setState({ phase: 'saving' })
        const assembledActions = actionResults.map(({ cfg, frames }) => ({
          id: crypto.randomUUID(),
          outfitId: resolvedOutfitId,
          name: cfg.name,
          kind: 'custom' as const,
          type: cfg.actionType,
          fps: fpsFromFrames(frames),
          keyFrameIndex: null,
          frames,
        }))
        const toUpdate: Character = {
          ...created,
          outfits: created.outfits.map((outfit, i) =>
            i === 0 ? { ...outfit, actions: assembledActions } : outfit,
          ),
        }
        await characterApis.update(toUpdate)

        // ── 步骤 6：完成
        setState({ phase: 'done', characterId, outfitId: resolvedOutfitId, message: null })
      }

      // 立刻启动管线，不阻塞调用方；任何异常收敛到 failed。
      run().catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error)
        fail(message || '快速创建过程中发生未知错误')
      })

      return {
        getState: () => state,
        subscribe(listener) {
          listeners.add(listener)
          return () => {
            listeners.delete(listener)
          }
        },
      }
    },
  }
}
