;WITH First (id, user_id, date, is_first, rn) AS
(
    SELECT *, ROW_NUMBER() OVER (ORDER BY user_id, date) rn FROM (
        SELECT
            id
            ,user_id
            ,date
            ,CASE WHEN ((LAG(log,1) OVER (ORDER BY user_id, date) <> log OR LAG(user_id,1) OVER (ORDER BY user_id, date) IS NULL) OR (LAG(user_id,1) OVER (ORDER BY user_id, date) <> user_id)) THEN 1 ELSE 0 END is_first
        FROM 
            (select date, user_id, log, row_number() over (order by date) as id from users_log)
    ) t
    WHERE t.is_first = 1
),
Last (id, user_id, date, is_last, rn) AS
(
	SELECT *, ROW_NUMBER() OVER (ORDER BY user_id, date) rn FROM (
		SELECT
			id
			,user_id
			,date
			, CASE 
				WHEN ((LEAD(log,1) OVER (ORDER BY user_id, date) <> log OR LEAD(log,1) OVER (ORDER BY user_id, date) IS NULL) OR (LEAD(user_id,1) OVER (ORDER BY user_id, date) <> user_id)) THEN 1 ELSE 0 END is_last
		FROM 
			(select date, user_id, log, row_number() over (order by date) as id from users_log)
	) t
	WHERE t.is_last = 1
)
SELECT
   c.user_id user_id, f.date start_date, l.date end_date, c.log log,  COUNT(*) length 
FROM
    First f
    LEFT JOIN Last l ON f.rn = l.rn
    LEFT JOIN (select date, user_id, log, row_number() over (order by date) as id from users_log) c ON c.user_id = f.user_id AND c.id BETWEEN f.id AND l.id
GROUP BY
    c.user_id, f.date, l.date, f.rn
ORDER BY 
    f.date, c.user_id


