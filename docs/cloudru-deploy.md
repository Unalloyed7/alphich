# Cloud.ru: автоматический GitHub Actions deploy после CI

Назначение: обновление уже работающего демостенда после успешного CI с сохранением локального Xray-прокси, `.env`, модели E5, индекса и базы данных на сервере.

Сначала проверьте, что `api`, `analysis-worker`, `frontend`, `postgres`, `rss-collector` и `worker` работают. `/home/alphich/.config/alphich/compose.vpn.yml` должен существовать; Xray запущен отдельно в Docker-сети `alphich-ai-proxy`. После успешного `CI — offline integration` для push в `main` автоматически запускается workflow `Deploy Cloud.ru (manual or after CI)`. Он повторно проверяет SHA текущего `main` и успешный push CI именно для этого SHA; устаревший автоматический запуск пропускается. Ручной запуск с `confirm=deploy` также сохранён.

## Конфигурация моделей демостенда

Сценарий объединяет `compose.yaml`, приватный `compose.vpn.yml`, затем публичный `compose.cloudru-ai.yaml`. Последний задаёт только настройки разрешённых моделей организаторов: Qwen для фактов, GLM для анализа, каталог JSON Schema и режим `gateway_receipt` с проверенным для этого шлюза коэффициентом `1`. Это не универсальный курс валют для другого API. Ключ остаётся в серверном `.env`, настройки прокси сохраняются, существующий денежный лимит и журнал расходов не меняются. В обоих контейнерах применяются одинаковые настройки. Автоматическая генерация выключена; проверочные карточки запускаются вручную.

При переходе со старого сценария, который ещё не подключал этот Compose-файл, после первого обновления исходников повторите deploy того же SHA вручную. Он запустит уже обновлённый сценарий с дополнительным файлом. Затем `/api/v1/llm/status` должен показывать `configured=true`, `budget_mode=gateway_receipt`; значение `prices_confirmed=false` допустимо для этого режима. Готовность настроек сама по себе не проверяет сетевое соединение: его подтверждает успешная задача анализа.

## Первоначальная подготовка

1. Сохраните эти файлы в GitHub, дождитесь зелёного `CI — offline integration`. Один раз обновите исходники на самом VPS командой `cd ~/alphich && git pull --ff-only origin main` (до первого GitHub Actions запуска), чтобы `scripts/deploy-cloudru.sh` был доступен.
2. На VPS создайте `/home/alphich/.config/alphich/ssh-deploy.sh` с содержимым ниже, затем `chmod 700 ~/.config/alphich/ssh-deploy.sh`. Каталог `~/.config/alphich` должен иметь `chmod 700`.

```bash
#!/usr/bin/env bash
set -euo pipefail
if [[ "${SSH_ORIGINAL_COMMAND:-}" =~ ^deploy\ ([0-9a-f]{40})$ ]]; then
  exec /usr/bin/bash /home/alphich/alphich/scripts/deploy-cloudru.sh "${BASH_REMATCH[1]}"
fi
echo 'Only deploy <commit-sha> is allowed' >&2
exit 1
```

3. На Mac создайте **отдельный** ключ без парольной фразы только для GitHub Actions: `ssh-keygen -t ed25519 -f ~/.ssh/alphich_github_deploy -N '' -C alphich-github-actions`. Приватный ключ нельзя отправлять в чат или коммитить.
4. Установите открытый ключ в `~/.ssh/authorized_keys` пользователя `alphich` с префиксом **`restrict,command="/home/alphich/.config/alphich/ssh-deploy.sh"`**. Это ограничивает новый ключ одной командой, не затрагивая существующий личный SSH-ключ Mac. Операцию удобно сделать через старое SSH-подключение:

```bash
cat ~/.ssh/alphich_github_deploy.pub | ssh -i ~/.ssh/alphich_cloudru -o IdentitiesOnly=yes alphich@89.232.176.172 \
  'umask 077; mkdir -p ~/.ssh; chmod 700 ~/.ssh; read -r pub; printf '\''restrict,command="/home/alphich/.config/alphich/ssh-deploy.sh" %s\n'\'' "$pub" >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys'
```

5. Проверьте подлинность host key VPS. На Mac сравните отпечатки `ssh-keyscan -t ed25519 89.232.176.172 2>/dev/null > /tmp/alphich-known-hosts && ssh-keygen -lf /tmp/alphich-known-hosts` и `ssh -i ~/.ssh/alphich_cloudru alphich@89.232.176.172 'ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub'`. Они должны совпасть. Не сохраняйте непроверенный host key.
6. В GitHub репозитории: **Settings → Secrets and variables → Actions → New repository secret**. Создайте `CLOUDRU_DEPLOY_KEY` (полный приватный ключ `~/.ssh/alphich_github_deploy`) и `CLOUDRU_KNOWN_HOSTS` (проверенная строка `/tmp/alphich-known-hosts`). Эти значения нельзя публиковать.
7. Для ручной проверки на GitHub в **Actions → Deploy Cloud.ru (manual or after CI) → Run workflow** выберите `main`, введите `deploy`. Сценарий перепроверит и пересоберёт контейнеры даже для уже установленного коммита. Обычные обновления идут через PR → merge в `main` → успешный push CI → автоматический deploy с резервной копией БД. Для такого обновления кода отдельный SSH-доступ разработчику не требуется; серверные секреты и VPN-настройки по-прежнему изменяются через доступ к самой VPS.

## Ограничения

- Скрипт не запускает платные LLM-вызовы и не меняет `AI_AUTOMATION_ENABLED`.
- Перед обновлением создаётся приватный `pg_dump -Fc` и контрольная сумма в `~/alphich/backups/`. Отдельно сохраняйте важные резервные копии **вне самой VPS**. Удаление VPS уничтожит и копии, хранящиеся на ней.
- Не используйте `docker compose down -v`.
- `create_all` не заменяет миграции БД. Изменения со сломом схемы требуют ручного плана миграции и восстановления. CI не гарантирует их безопасность.
- При ошибке деплоя **нет автоматического отката** схемы или данных. Разберите ошибку и используйте резервную копию лишь по отдельной процедуре восстановления.
- Обновления GitHub применяются через `-f compose.yaml -f ~/.config/alphich/compose.vpn.yml -f compose.cloudru-ai.yaml --profile news`, поэтому VPN-настройка сохраняется и RSS продолжает работать.
- Пароли, ключи, модель E5, PDF, индекс и состояние PostgreSQL исключены из Git; команда `git pull` их не переносит.
- Права workflow позволяют читать репозиторий и статусы CI. SSH-ключ ограничен forced command, но код, развёртываемый на сервер, должен проходить review: пользователь `alphich` имеет доступ к `sudo docker`.
