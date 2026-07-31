import type { WorkflowRevision, WorkflowStep } from '@/entities'

export { createWorkflowController } from './controller'
export type {
  CreateWorkflowControllerInput,
  CreateWorkflowControllerOptions,
  WorkflowController,
} from './controller'

/** 更新当前 Revision 中某个步骤的业务数据。 */
export interface UpdateWorkflowStepInput {
  stepId: WorkflowStep['id']
  data: unknown
}

/** 从指定 Revision 的指定步骤建立新的执行版本。 */
export interface RestartWorkflowFromStepInput {
  revisionId: WorkflowRevision['id']
  stepId: WorkflowStep['id']
}

/** 把某次服务端调用的结果写回目标步骤。 */
export interface ApplyServerResultInput {
  /** 发起请求时所属的 Revision，防止旧的异步结果污染重启后的新版本。 */
  revisionId: WorkflowRevision['id']
  stepId: WorkflowStep['id']
  result: unknown
}
