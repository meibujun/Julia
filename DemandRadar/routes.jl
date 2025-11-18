using Genie.Router
using .DashboardController

route("/") do
  serve_static_file("welcome.html")
end

route("/dashboard", DashboardController.index)