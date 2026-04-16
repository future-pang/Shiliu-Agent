from typing import Dict
from server.agent.state import AgentState, TaskPlan

def task_fetcher_node(state: AgentState) -> dict:
    """
    [核心调度中枢]：负责解析 DAG 依赖、前置上下文注入、并推拉状态机。
    它的输出会自动被 graph.py 中的 dispatch_tasks_from_fetcher 路由函数读取，从而触发并发。
    """
    print("\n[Fetcher]开始扫描任务队列状态...")

    # ==== 遍历所有 pending 任务 ====
    tasks: Dict[int, TaskPlan] = state.get("tasks", {})
    if not tasks:
        return {}
    updates: Dict[int, TaskPlan] = {}
    for step_id, task in tasks.items():
        if task.get("status") != "pending":
            continue

        # ==== 检查依赖是否满足 ====
        depends_on = task.get("depends_on", [])
        is_ready = True
        dependency_results = []

        for dep_id in depends_on:
            dep_task = tasks.get(dep_id)
            if not dep_task:
                print(f"[Fetcher] 警告: 任务 {step_id} 依赖了不存在的任务 {dep_id}，自动忽略该依赖。")
                continue

            if dep_task.get("status") != "done":
                is_ready = False
                break
            else:
                res = dep_task.get("result", "无明确结果返回")
                dependency_results.append(f"前置步骤 [{dep_id}] ({dep_task['description']}) 的结果:\n{res}")

        # ==== 如果依赖全部满足 ====
        if is_ready:
            print(f"[Fetcher] 任务 {step_id} 依赖已满足，准备发车！")

            # 拷贝一份当前任务，准备更新
            updated_task = task.copy()
            updated_task["status"] = "running"

            # 前置知识注入
            if dependency_results:
                context_str = "\n\n".join(dependency_results)
                enhanced_description = (
                    f"{task['description']}\n\n"
                    f"========== [系统注入：前置任务执行结果参考] ==========\n"
                    f"{context_str}\n"
                    f"======================================================\n"
                    f"请基于上述前置信息，完成你的任务。"
                )
                updated_task["description"] = enhanced_description
                print(f"      -> 已为任务 {step_id} 注入 {len(dependency_results)} 个前置结果上下文。")
            updates[step_id] = updated_task

    if updates:
        print(f"[Fetcher]本轮放行 {len(updates)} 个任务进入执行池。")
    else:
        print("[Fetcher]当前无新任务就绪，等待正在执行的任务返回...")
    return {"tasks": updates}

# =====================================================================
# 极端异常兜底：死锁熔断器
# =====================================================================
def deadlock_resolver_node(state: AgentState) -> dict:
    """
    当 graph.py 里的路由函数发现"没有任务在跑，但还有 pending 任务"时，就会触发这里。
    这通常意味着 Planner 大模型产生了幻觉，写出了循环依赖 (A->B, B->A)。
    策略：暴力解开所有 pending 任务的 depends_on 枷锁，强行降级为全并发模式。
    """
    print("\n[Deadlock Resolver] 触发死锁熔断机制！正在强行解开循环依赖...")
    tasks: Dict[int, TaskPlan] = state.get("tasks", {})
    updates: Dict[int, TaskPlan] = {}

    for step_id, task in tasks.items():
        if task.get("status") == "pending":
            updated_task = task.copy()
            updated_task["depends_on"] = []     # 强制清空依赖
            updated_task["status"] = "running"  # 强行起步

            # 警告
            updated_task["description"] += "\n(注：此任务因依赖死锁被系统强制并发执行，如果缺少必要参数，请尽力而为或说明缺乏前提信息。)"

            updates[step_id] = updated_task
            print(f"已强制解锁并放行任务: {step_id}")

    return {"tasks": updates}











